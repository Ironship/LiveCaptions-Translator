using System.IO;
using NAudio.Wave;
using Whisper.net;
using Whisper.net.Ggml;

namespace LiveCaptionsTranslator.utils
{
    // Fully offline speech recognition using the OpenAI Whisper model family
    // (whisper.cpp via Whisper.net). Supports Small (244 MB) and Medium (769 MB)
    // model sizes — both offer excellent German accuracy at different speed/size tradeoffs.
    //
    // Audio is captured from the microphone, segmented via voice-activity detection (VAD),
    // and each utterance is transcribed locally by the Whisper engine.
    public class WhisperSTTHandler : IDisposable
    {
        private const int SAMPLE_RATE = 16000;
        private const int BITS_PER_SAMPLE = 16;
        private const int CHANNELS = 1;

        // VAD thresholds — tune SILENCE_THRESHOLD_RMS for microphone sensitivity.
        private const double SILENCE_THRESHOLD_RMS = 400.0;
        private const double SILENCE_END_SEC = 0.8;
        private const double MAX_CHUNK_SEC = 15.0;
        private const double MIN_SPEECH_SEC = 0.3;
        private static readonly int MinSpeechSamples = (int)(MIN_SPEECH_SEC * SAMPLE_RATE);

        private const int MAX_HISTORY = 5;

        private WaveInEvent? waveIn;
        private MemoryStream audioStream = new();
        private WaveFileWriter? audioWriter;

        private WhisperFactory? whisperFactory;
        private WhisperProcessor? processor;

        private readonly Queue<string> history = new();
        private string partial = string.Empty;
        private readonly object syncLock = new();

        // VAD counters (accessed only from the audio-data callback thread).
        private int silentSamples;
        private int speechSamples;
        private int totalSamplesInChunk;
        private bool hasSpeech;
        private bool isTranscribing;

        private bool disposed;

        public bool IsRunning { get; private set; }
        public string? LastError { get; private set; }

        // Returns the default file name stored next to the executable for a given model size.
        public static string GetDefaultModelPath(GgmlType ggmlType) =>
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
                $"ggml-{ggmlType.ToString().ToLower()}.bin");

        // Returns a text block that mirrors the format returned by LiveCaptions:
        // completed utterances followed by "..." while a chunk is being transcribed.
        public string GetCaptions()
        {
            lock (syncLock)
            {
                string hist = string.Join(" ", history);
                if (string.IsNullOrEmpty(partial))
                    return hist;
                string sep = hist.Length > 0 ? " " : string.Empty;
                return hist + sep + partial;
            }
        }

        // Starts audio capture and loads the Whisper model.
        // modelPath — full path to a ggml-*.bin model file; if empty, uses the default path.
        // ggmlType  — determines the default model file name when modelPath is empty.
        // language  — BCP-47 language code passed to Whisper (e.g. "de"); defaults to German.
        public bool TryStart(string modelPath, GgmlType ggmlType, string language = "de")
        {
            if (IsRunning)
                return true;

            if (string.IsNullOrWhiteSpace(modelPath))
                modelPath = GetDefaultModelPath(ggmlType);

            if (!File.Exists(modelPath))
            {
                int sizeMB = GetApproxModelSizeMB(ggmlType);
                LastError = $"Whisper model not found at \"{modelPath}\" (~{sizeMB} MB). " +
                            "Use the Download button in Settings to fetch it automatically, " +
                            "or place the ggml-*.bin file there manually.";
                return false;
            }

            try
            {
                whisperFactory = WhisperFactory.FromPath(modelPath);
                processor = whisperFactory.CreateBuilder()
                    .WithLanguage(language)
                    .WithNoContext()
                    .Build();

                BeginNewChunk();

                waveIn = new WaveInEvent
                {
                    WaveFormat = new WaveFormat(SAMPLE_RATE, BITS_PER_SAMPLE, CHANNELS)
                };
                waveIn.DataAvailable += OnDataAvailable;
                waveIn.StartRecording();

                IsRunning = true;
                LastError = null;
                return true;
            }
            catch (Exception ex)
            {
                LastError = ex.Message;
                DisposeInternals();
                return false;
            }
        }

        // Downloads the GGML model file from Hugging Face to targetPath.
        // Returns false (sets LastError) if the download fails.
        public async Task<bool> DownloadModelAsync(GgmlType ggmlType, string targetPath)
        {
            try
            {
                string dir = Path.GetDirectoryName(targetPath)
                    ?? AppDomain.CurrentDomain.BaseDirectory;
                Directory.CreateDirectory(dir);

                using var modelStream =
                    await WhisperGgmlDownloader.GetGgmlModelAsync(ggmlType).ConfigureAwait(false);
                using var fileStream = File.Create(targetPath);
                await modelStream.CopyToAsync(fileStream).ConfigureAwait(false);

                LastError = null;
                return true;
            }
            catch (Exception ex)
            {
                LastError = ex.Message;
                return false;
            }
        }

        public void Stop()
        {
            if (!IsRunning)
                return;

            try { waveIn?.StopRecording(); } catch { }

            lock (syncLock)
            {
                history.Clear();
                partial = string.Empty;
            }

            IsRunning = false;
        }

        private void BeginNewChunk()
        {
            audioWriter?.Dispose();
            audioStream.Dispose();
            audioStream = new MemoryStream();
            audioWriter = new WaveFileWriter(
                audioStream, new WaveFormat(SAMPLE_RATE, BITS_PER_SAMPLE, CHANNELS));
            silentSamples = 0;
            speechSamples = 0;
            totalSamplesInChunk = 0;
            hasSpeech = false;
        }

        private void OnDataAvailable(object? sender, WaveInEventArgs e)
        {
            try
            {
                audioWriter?.Write(e.Buffer, 0, e.BytesRecorded);

                double rms = CalculateRms(e.Buffer, e.BytesRecorded);
                int samplesInBlock = e.BytesRecorded / 2; // 16-bit → 2 bytes per sample
                totalSamplesInChunk += samplesInBlock;

                if (rms >= SILENCE_THRESHOLD_RMS)
                {
                    hasSpeech = true;
                    speechSamples += samplesInBlock;
                    silentSamples = 0;
                }
                else if (hasSpeech)
                {
                    silentSamples += samplesInBlock;
                    double silentSec = (double)silentSamples / SAMPLE_RATE;
                    double totalSec = (double)totalSamplesInChunk / SAMPLE_RATE;

                    bool endBySilence = silentSec >= SILENCE_END_SEC;
                    bool endByLength = totalSec >= MAX_CHUNK_SEC;

                    if ((endBySilence || endByLength) &&
                        speechSamples >= MinSpeechSamples &&
                        !isTranscribing)
                    {
                        SubmitChunk();
                    }
                }
            }
            catch { }
        }

        private void SubmitChunk()
        {
            try
            {
                audioWriter?.Flush();
                byte[] wavBytes = audioStream.ToArray();

                BeginNewChunk();

                isTranscribing = true;
                lock (syncLock)
                    partial = "...";

                _ = Task.Run(async () =>
                {
                    try { await TranscribeAsync(wavBytes).ConfigureAwait(false); }
                    finally { isTranscribing = false; }
                });
            }
            catch { }
        }

        private async Task TranscribeAsync(byte[] wavBytes)
        {
            try
            {
                using var stream = new MemoryStream(wavBytes);
                var segments = new List<string>();

                await foreach (var seg in processor!.ProcessAsync(stream).ConfigureAwait(false))
                {
                    string t = seg.Text?.Trim() ?? string.Empty;
                    if (!string.IsNullOrWhiteSpace(t))
                        segments.Add(t);
                }

                string text = string.Join(" ", segments).Trim();

                lock (syncLock)
                {
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        history.Enqueue(text.TrimEnd('.', ' ') + ".");
                        while (history.Count > MAX_HISTORY)
                            history.Dequeue();
                    }
                    partial = string.Empty;
                }
            }
            catch (Exception ex)
            {
                lock (syncLock)
                    partial = string.Empty;
                LastError = ex.Message;
            }
        }

        private static double CalculateRms(byte[] buffer, int bytesRecorded)
        {
            if (bytesRecorded < 2)
                return 0.0;

            double sumSq = 0;
            int samples = bytesRecorded / 2;
            for (int i = 0; i + 1 < bytesRecorded; i += 2)
            {
                short s = (short)(buffer[i] | (buffer[i + 1] << 8));
                sumSq += (double)s * s;
            }
            return Math.Sqrt(sumSq / samples);
        }

        private static int GetApproxModelSizeMB(GgmlType t) => t switch
        {
            GgmlType.Small => 244,
            GgmlType.Medium => 769,
            _ => 0,
        };

        private void DisposeInternals()
        {
            try { waveIn?.Dispose(); } catch { }
            try { audioWriter?.Dispose(); } catch { }
            try { audioStream.Dispose(); } catch { }
            try { processor?.Dispose(); } catch { }
            try { whisperFactory?.Dispose(); } catch { }
            waveIn = null;
            audioWriter = null;
            processor = null;
            whisperFactory = null;
        }

        public void Dispose()
        {
            if (disposed)
                return;
            Stop();
            DisposeInternals();
            disposed = true;
        }
    }
}
