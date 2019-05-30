source config.sh
echo $ZONE
gcloud compute instances create $INSTANCE_NAME --project=$PROJECT --zone=$ZONE --image-family=$IMAGE_FAMILY --image-project=deeplearning-platform-release --maintenance-policy=TERMINATE --accelerator=$INSTANCE_SPEC --metadata="install-nvidia-driver=True" --boot-disk-size=120GB --maintenance-policy=TERMINATE --machine-type=n1-standard-8
