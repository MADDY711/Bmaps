package com.visioncompanion.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
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
import kotlinx.coroutines.launch
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import java.io.File
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {

    // ─── Views ────────────────────────────────────────────────────────────────
    private lateinit var previewView: PreviewView
    private lateinit var statusText: TextView
    private lateinit var analyzeButton: Button
    private lateinit var chatButton: Button

    // ─── Camera ───────────────────────────────────────────────────────────────
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService

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

        // Init TTS
        tts = TextToSpeech(this, this)

        // Init camera thread
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Button listeners
        analyzeButton.setOnClickListener { captureAndAnalyze() }
        chatButton.setOnClickListener    { sendTestChat() }

        // Permissions → camera
        if (hasPermissions()) {
            startCamera()
            checkServerConnection()
        } else {
            ActivityCompat.requestPermissions(this, PERMISSIONS, REQUEST_PERMISSIONS)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tts?.stop()
        tts?.shutdown()
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
            checkServerConnection()
        } else {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
        }
    }

    // ─── Camera ───────────────────────────────────────────────────────────────

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

            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture
                )
                setStatus("📷 Camera ready")
            } catch (e: Exception) {
                Log.e(TAG, "Camera failed: ${e.message}")
                setStatus("❌ Camera error: ${e.message}")
            }

        }, ContextCompat.getMainExecutor(this))
    }

    // ─── Server Connection Test ───────────────────────────────────────────────

    private fun checkServerConnection() {
        lifecycleScope.launch {
            setStatus("🔄 Connecting to server...")
            try {
                val response = ApiClient.api.healthCheck()
                if (response.isSuccessful) {
                    setStatus("✅ Server connected! Ready.")
                    speak("Vision Companion is ready")
                } else {
                    setStatus("⚠️ Server error: ${response.code()}")
                }
            } catch (e: Exception) {
                setStatus("❌ Cannot reach server. Check IP in ApiClient.kt")
                Log.e(TAG, "Server check failed: ${e.message}")
            }
        }
    }

    // ─── Analyze Scene ────────────────────────────────────────────────────────

    private fun captureAndAnalyze() {
        val capture = imageCapture ?: run {
            setStatus("❌ Camera not ready")
            return
        }

        setStatus("📸 Capturing...")
        analyzeButton.isEnabled = false

        val file = File(externalMediaDirs.firstOrNull(), "${System.currentTimeMillis()}.jpg")

        capture.takePicture(
            ImageCapture.OutputFileOptions.Builder(file).build(),
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    uploadForAnalysis(file)
                }
                override fun onError(e: ImageCaptureException) {
                    setStatus("❌ Capture failed: ${e.message}")
                    analyzeButton.isEnabled = true
                }
            }
        )
    }

    private fun uploadForAnalysis(file: File) {
        lifecycleScope.launch {
            setStatus("🔍 Analyzing scene...")
            try {
                val part     = ApiClient.createImagePart(file)
                val response = ApiClient.api.analyzeScene(part, "general")

                if (response.isSuccessful) {
                    val desc = response.body()?.description ?: "No description"
                    setStatus("👁️ $desc")
                    speak(desc)
                } else {
                    setStatus("❌ Analysis failed: ${response.code()}")
                    speak("Analysis failed")
                }
            } catch (e: Exception) {
                setStatus("❌ Error: ${e.message}")
                Log.e(TAG, "Analysis error", e)
            } finally {
                file.delete()
                analyzeButton.isEnabled = true
            }
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