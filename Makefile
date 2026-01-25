all: list_cams.exe realtime_video_pipeline.exe

list_cams.exe: list_cams.cpp
	g++ -std=c++20 -o list_cams.exe list_cams.cpp -I /opt/homebrew/include/opencv4 -L /opt/homebrew/lib -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui

realtime_video_pipeline.exe: realtime_video_pipeline.cpp
	g++ -std=c++20 -o realtime_video_pipeline.exe realtime_video_pipeline.cpp -I ../openai-cpp/include/openai -I /opt/homebrew/include/opencv4 -L /opt/homebrew/lib -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs -lcurl

clean:
	rm -f *.exe	


	