mkdir -p weights
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1DJQhQHFBaESIToLsMRPNvtbsoBl4F_yh' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DJQhQHFBaESIToLsMRPNvtbsoBl4F_yh" -O "weights/arboles512.pkl" && rm -rf /tmp/cookies.txt


https://drive.google.com/file/d/1DJQhQHFBaESIToLsMRPNvtbsoBl4F_yh/view?usp=sharing