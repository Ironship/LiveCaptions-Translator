using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using NAudio.Wave;

namespace LiveCaptionsTranslator.utils
{
    // Offline speech recognition using a locally-running Voxtral inference server
    // (e.g. vLLM: `vllm serve mistralai/Voxtral-Mini-3B-2507`).
    //
    // The server exposes the same multipart audio/transcriptions endpoint as the
    // Mistral cloud API, so the same HTTP client works for both local and cloud use.
    // The default URL points to localhost (vLLM default port 8000).
    // Set ApiKey to any non-empty string for local servers that don't require auth.
    //
    // Compatible with voxtral-mini-2507 (VoxtralMini) and voxtral-small-2507 (VoxtralSmall).
    public class VoxtralSTTHandler : IDisposable
    {
        private const int SAMPLE_RATE = 16000;
        private const int BITS_PER_SAMPLE = 16;
        private const int CHANNELS = 1;

        private const double SILENCE_THRESHOLD_RMS = 400.0;
        private const double SILENCE_END_SEC = 0.8;
        private const double MAX_CHUNK_SEC = 15.0;
        private const double MIN_SPEECH_SEC = 0.3;
        private static readonly int MinSpeechSamples = (int)(MIN_SPEECH_SEC * SAMPLE_RATE);

        private const int MAX_HISTORY = 5;

        public const string DEFAULT_API_URL = "http://localhost:8000/v1";

        private WaveInEvent? waveIn;
        private MemoryStream audioStream = new();
        private WaveFileWriter? audioWriter;
        private readonly HttpClient httpClient = new();

        private string apiKey = string.Empty;
        private string apiUrl = string.Empty;
        private string model = string.Empty;
        private string language = string.Empty;

        private readonly Queue<string> history = new();
        private string partial = string.Empty;
        private readonly object syncLock = new();

        private int silentSamples;
        private int speechSamples;
        private int totalSamplesInChunk;
        private bool hasSpeech;
        private bool isTranscribing;

        private bool disposed;

        public bool IsRunning { get; private set; }
        public string? LastError { get; private set; }

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

        public bool TryStart(string apiUrl, string model, string apiKey = "", string language = "de")
        {
            if (IsRunning)
                return true;

            if (string.IsNullOrWhiteSpace(apiUrl))
                apiUrl = DEFAULT_API_URL;

            try
            {
                this.apiKey = apiKey;
                this.apiUrl = apiUrl.TrimEnd('/');
                this.model = model;
                this.language = language;

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
                int samplesInBlock = e.BytesRecorded / 2;
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

                    if ((silentSec >= SILENCE_END_SEC || totalSec >= MAX_CHUNK_SEC) &&
                        speechSamples >= MinSpeechSamples && !isTranscribing)
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
                using var content = new MultipartFormDataContent();
                var fileContent = new ByteArrayContent(wavBytes);
                fileContent.Headers.ContentType = new MediaTypeHeaderValue("audio/wav");
                content.Add(fileContent, "file", "audio.wav");
                content.Add(new StringContent(model), "model");
                if (!string.IsNullOrWhiteSpace(language))
                    content.Add(new StringContent(language), "language");

                using var request = new HttpRequestMessage(
                    HttpMethod.Post, $"{apiUrl}/audio/transcriptions");
                if (!string.IsNullOrWhiteSpace(apiKey))
                    request.Headers.Add("x-api-key", apiKey);
                request.Content = content;

                using var response = await httpClient.SendAsync(request).ConfigureAwait(false);
                response.EnsureSuccessStatusCode();

                var json = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
                using var doc = JsonDocument.Parse(json);
                string text = (doc.RootElement.GetProperty("text").GetString() ?? string.Empty).Trim();

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

        private void DisposeInternals()
        {
            try { waveIn?.Dispose(); } catch { }
            try { audioWriter?.Dispose(); } catch { }
            try { audioStream.Dispose(); } catch { }
            try { httpClient.Dispose(); } catch { }
            waveIn = null;
            audioWriter = null;
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
