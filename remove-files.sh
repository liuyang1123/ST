sudo find . -type d -name "__pycache__" -exec rm -r "{}" \;
sudo find . -type d -name "migrations" -exec rm -r "{}" \;
sudo find . -type f -name "*.pyc" -delete;
sudo find . -type f -name "*.py~" -delete;
