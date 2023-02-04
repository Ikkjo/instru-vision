cd ../models/tfjs/
python ../../notebooks/utils/cors_fileserver.py &
cd ../../js/guitar-classification-app
serve -s build
