from google.cloud import aiplatform

PROJECT_ID = "your-project-id"
REGION = "us-central1"

aiplatform.init(project=PROJECT_ID, location=REGION)

model = aiplatform.Model.upload(
    display_name="ml-ai-xgboost-model",
    artifact_uri="models/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-5:latest",
)

print(model.resource_name)