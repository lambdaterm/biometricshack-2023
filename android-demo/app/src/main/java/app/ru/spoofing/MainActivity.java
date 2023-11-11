package app.ru.spoofing;


import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.util.List;

import app.ru.spoofing.ml.ModelBig;

public class MainActivity extends AppCompatActivity {

    Button camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize = 224;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, 3);
                } else {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });
        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(cameraIntent, 1);
            }
        });
    }

    @SuppressLint("DefaultLocale")
    public void classifyImage(Bitmap image) {
//        try {


        FaceDetectorOptions highAccuracyOpts =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                        .build();

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .add(new CastOp(DataType.FLOAT32))
                .build();

        FaceDetector detector = FaceDetection.getClient(highAccuracyOpts);
        Task<List<Face>> face_result_task = detector.process(InputImage.fromBitmap(image, 0));

        face_result_task.addOnSuccessListener(
                new OnSuccessListener<List<Face>>() {
                    @Override
                    public void onSuccess(List<Face> faces) {
                        if (!faces.isEmpty()) {
                            System.out.println(String.format("faces count: %d", faces.size()));
                            try {
                                ModelBig model = ModelBig.newInstance(getApplicationContext());
                                TensorImage tensorImage = imageProcessor.process(TensorImage.fromBitmap(image));

                                ModelBig.Outputs outputs = model.process(tensorImage.getTensorBuffer());
                                TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                                float[] confidences = outputFeature0.getFloatArray();

                                float current_conf = confidences[0];

                                if (current_conf > 0.5) {
                                    result.setTextColor(getResources().getColor(R.color.green));
                                    result.setText(String.format("REAL: %f", current_conf));
                                } else {
                                    result.setTextColor(getResources().getColor(R.color.red));
                                    result.setText(String.format("FAKE: %f", current_conf));
                                }
                                model.close();
                            } catch (IOException e) {
                                //
                            }
                        } else {
                            System.out.println(String.format("faces count: %d", faces.size()));
                            result.setTextColor(getResources().getColor(R.color.red));
                            result.setText("No faces found. Skip classification");
                        }
                    }
                }
        );

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == 3) {
                Bitmap image = (Bitmap) data.getExtras().get("data");
                assert image != null;

                Glide.with(this).load(image).into(imageView);

                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);
                classifyImage(image);
            } else {
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                Glide.with(this).load(image).into(imageView);

                assert image != null;
                image = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);

                classifyImage(image);
            }
        }
        super.onActivityResult(requestCode, resultCode, data);
    }
}