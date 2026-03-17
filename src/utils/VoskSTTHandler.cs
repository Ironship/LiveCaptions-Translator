using System.IO;
using System.Text.Json;
using NAudio.Wave;
using Vosk;

namespace LiveCaptionsTranslator.utils
{
    // Offline speech recognition engine using Vosk.
    // Produces a rolling window of recognized text to mimic the LiveCaptions text format.
    public class VoskSTTHandler : IDisposable
    {
        // Keep the last 5 completed utterances so the translation pipeline
        // always has context for sentence-boundary detection.
        private const int MAX_HISTORY = 5;

        // Vosk recognises audio at 16 kHz mono.
        private const float SAMPLE_RATE = 16000.0f;

        private Model? model;
        private VoskRecognizer? recognizer;
        private WaveInEvent? waveIn;

        private readonly Queue<string> history = new();
        private string partial = string.Empty;
        private readonly object syncLock = new();

        private bool disposed = false;

        public bool IsRunning { get; private set; } = false;
        public string? LastError { get; private set; }

        // Suggested path where users should place the downloaded Vosk model folder.
        public static string DefaultModelPath =>
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "vosk-model");

        // Returns a text block similar to what LiveCaptions produces:
        // completed utterances followed by the current partial result.
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

        // Starts the Vosk recognizer and the audio capture device.
        // Returns false (and sets LastError) if the model path is invalid or startup fails.
        public bool TryStart(string modelPath)
        {
            if (IsRunning)
                return true;

            if (string.IsNullOrWhiteSpace(modelPath) || !Directory.Exists(modelPath))
            {
                LastError = $"Vosk model directory not found: \"{modelPath}\". " +
                            "Download a model from https://alphacephei.com/vosk/models " +
                            "and set the path in Settings.";
                return false;
            }

            try
            {
                Vosk.Vosk.SetLogLevel(-1); // suppress verbose native logging
                model = new Model(modelPath);
                recognizer = new VoskRecognizer(model, SAMPLE_RATE);
                recognizer.SetMaxAlternatives(0);
                recognizer.SetWords(false);

                waveIn = new WaveInEvent
                {
                    WaveFormat = new WaveFormat((int)SAMPLE_RATE, 1)
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

            try
            {
                waveIn?.StopRecording();
            }
            catch { }

            lock (syncLock)
            {
                history.Clear();
                partial = string.Empty;
            }

            IsRunning = false;
        }

        private void OnDataAvailable(object? sender, WaveInEventArgs e)
        {
            if (recognizer == null)
                return;

            try
            {
                if (recognizer.AcceptWaveform(e.Buffer, e.BytesRecorded))
                {
                    // Final result for a complete utterance.
                    using var doc = JsonDocument.Parse(recognizer.Result());
                    string text = doc.RootElement.GetProperty("text").GetString() ?? string.Empty;
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        lock (syncLock)
                        {
                            history.Enqueue(text + ".");
                            while (history.Count > MAX_HISTORY)
                                history.Dequeue();
                            partial = string.Empty;
                        }
                    }
                }
                else
                {
                    // Partial (in-progress) result — updated every audio chunk.
                    using var doc = JsonDocument.Parse(recognizer.PartialResult());
                    string text = doc.RootElement.GetProperty("partial").GetString() ?? string.Empty;
                    lock (syncLock)
                        partial = text;
                }
            }
            catch { }
        }

        private void DisposeInternals()
        {
            try { waveIn?.Dispose(); } catch { }
            try { recognizer?.Dispose(); } catch { }
            try { model?.Dispose(); } catch { }
            waveIn = null;
            recognizer = null;
            model = null;
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
