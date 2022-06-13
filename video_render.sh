conda deactivate
ffmpeg -r 10 -i "%04d.png" -vcodec libx264 "output.mp4"
conda activate nice-slam