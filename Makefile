bin/output.mp4:
	ffmpeg -framerate 5 -i frames/frame%03d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p bin/output.mp4
