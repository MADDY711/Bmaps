package com.visioncompanion.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.media.*
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Base64
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener, GeminiLiveClient.GeminiLiveListener {

    // ─── Views ────────────────────────────────────────────────────────────────
    private lateinit var previewView: PreviewView
    private lateinit var statusText: TextView
    private lateinit var analyzeButton: Button
    private lateinit var chatButton: Button

    // ─── Camera ───────────────────────────────────────────────────────────────
    private var imageCapture: ImageCapture? = null
    private var imageAnalysis: ImageAnalysis? = null
    private lateinit var cameraExecutor: ExecutorService

    // ─── Gemini Live ─────────────────────────────────────────────────────────
    private lateinit var geminiClient: GeminiLiveClient
    private var isStreaming = false
    private var lastFrameTime = 0L

    // ─── Audio (Capture) ─────────────────────────────────────────────────────
    private var audioRecord: AudioRecord? = null
    private var audioCaptureJob: Job? = null
    private val SAMPLE_RATE = 16000
    private val BUFFER_SIZE = AudioRecord.getMinBufferSize(
        SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
    )

    // ─── Audio (Playback) ────────────────────────────────────────────────────
    private var audioTrack: AudioTrack? = null
    private val PLAYBACK_RATE = 24000 // Gemini returns 24kHz

    // ─── TTS ──────────────────────────────────────────────────────────────────
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    companion object {
        private const val TAG = "VisionCompanion"
        private const val REQUEST_PERMISSIONS = 10
        private val PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )
    }

    // ─── Lifecycle ────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Bind views
        previewView   = findViewById(R.id.previewView)
        statusText    = findViewById(R.id.statusText)
        analyzeButton = findViewById(R.id.analyzeButton)
        chatButton    = findViewById(R.id.chatButton)

        // Init Gemini Client
        geminiClient = GeminiLiveClient(ApiClient.WS_URL, this)

        // Init TTS (Keep for fallback/notifications)
        tts = TextToSpeech(this, this)

        // Init camera thread
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Init AudioTrack for playback
        initAudioTrack()

        // Button listeners
        analyzeButton.setOnClickListener { 
            if (!isStreaming) startLiveStreaming() else stopLiveStreaming()
        }
        chatButton.setOnClickListener { sendTestChat() }

        // Permissions → camera
        if (hasPermissions()) {
            startCamera()
            geminiClient.connect()
        } else {
            ActivityCompat.requestPermissions(this, PERMISSIONS, REQUEST_PERMISSIONS)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopLiveStreaming()
        cameraExecutor.shutdown()
        geminiClient.disconnect()
        audioTrack?.release()
        tts?.stop()
        tts?.shutdown()
    }

    // ─── Gemini Live Callbacks ───────────────────────────────────────────────

    override fun onConnected() {
        setStatus("✅ Live Session Connected")
    }

    override fun onDisconnected() {
        setStatus("❌ Live Session Disconnected")
        stopLiveStreaming()
    }

    override fun onError(message: String) {
        setStatus("⚠️ Gemini Error: $message")
    }

    override fun onAudioDataReceived(base64Audio: String) {
        try {
            val audioData = Base64.decode(base64Audio, Base64.DEFAULT)
            audioTrack?.write(audioData, 0, audioData.size)
            audioTrack?.play()
        } catch (e: Exception) {
            Log.e(TAG, "Playback error: ${e.message}")
        }
    }

    override fun onTranscriptReceived(text: String) {
        setStatus("🤖 $text")
    }

    override fun onTurnComplete() {
        setStatus("👂 Listening...")
    }

    // ─── Streaming Logic ─────────────────────────────────────────────────────

    private fun startLiveStreaming() {
        if (!hasPermissions()) return
        
        isStreaming = true
        analyzeButton.text = "Stop Live"
        setStatus("🚀 Starting Live Stream...")

        // 1. Start Audio Capture
        startAudioCapture()
    }

    private fun stopLiveStreaming() {
        isStreaming = false
        analyzeButton.text = "Start Live"
        setStatus("🛑 Stream Stopped")
        
        audioCaptureJob?.cancel()
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }

    private fun startAudioCapture() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            BUFFER_SIZE
        )

        audioRecord?.startRecording()
        
        audioCaptureJob = lifecycleScope.launch(Dispatchers.IO) {
            val buffer = ByteArray(BUFFER_SIZE)
            while (isStreaming) {
                val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                if (read > 0) {
                    val base64 = Base64.encodeToString(buffer, 0, read, base64.NO_WRAP)
                    geminiClient.sendAudio(base64)
                }
            }
        }
    }

    private fun initAudioTrack() {
        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build())
            .setAudioFormat(AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setSampleRate(PLAYBACK_RATE)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build())
            .setBufferSizeInBytes(BUFFER_SIZE)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
    }

    // ─── CameraX ───────────────────────────────────────────────────────────────

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            // Video Analysis use-case (1 FPS)
            imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        if (isStreaming) {
                            val currentTime = System.currentTimeMillis()
                            if (currentTime - lastFrameTime > 2000) { // Send every 2 seconds
                                processImageProxy(imageProxy)
                                lastFrameTime = currentTime
                            } else {
                                imageProxy.close()
                            }
                        } else {
                            imageProxy.close()
                        }
                    }
                }

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture,
                    imageAnalysis
                )
                setStatus("📷 Camera ready")
            } catch (e: Exception) {
                Log.e(TAG, "Camera failed: ${e.message}")
                setStatus("❌ Camera error: ${e.message}")
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImageProxy(image: ImageProxy) {
        try {
            val bitmap = imageProxyToBitmap(image)
            val outputStream = ByteArrayOutputStream()
            bitmap?.compress(Bitmap.CompressFormat.JPEG, 50, outputStream)
            val base64 = Base64.encodeToString(outputStream.toByteArray(), base64.NO_WRAP)
            geminiClient.sendVideoFrame(base64)
        } catch (e: Exception) {
            Log.e(TAG, "Image processing failed: ${e.message}")
        } finally {
            image.close()
        }
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    // ─── Permissions ─────────────────────────────────────────────────────────

    private fun hasPermissions() = PERMISSIONS.all {
        ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_PERMISSIONS && hasPermissions()) {
            startCamera()
            geminiClient.connect()
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
        }
    }

    // ─── Test Chat ────────────────────────────────────────────────────────────

    private fun sendTestChat() {
        lifecycleScope.launch {
            setStatus("💬 Sending message...")
            chatButton.isEnabled = false
            try {
                val msgBody    = "Hello! I need help navigating my surroundings."
                    .toRequestBody("text/plain".toMediaTypeOrNull())
                val userBody   = "android_user"
                    .toRequestBody("text/plain".toMediaTypeOrNull())

                val response = ApiClient.api.chat(msgBody, userBody)

                if (response.isSuccessful) {
                    val reply = response.body()?.response ?: "No response"
                    setStatus("🤖 $reply")
                    speak(reply)
                } else {
                    setStatus("❌ Chat failed: ${response.code()}")
                }
            } catch (e: Exception) {
                setStatus("❌ Error: ${e.message}")
                Log.e(TAG, "Chat error", e)
            } finally {
                chatButton.isEnabled = true
            }
        }
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    private fun setStatus(message: String) {
        runOnUiThread { statusText.text = message }
        Log.d(TAG, message)
    }

    private fun speak(text: String) {
        if (ttsReady) {
            tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, UUID.randomUUID().toString())
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts?.language = Locale.US
            ttsReady = true
            Log.d(TAG, "TTS ready")
        }
    }
}