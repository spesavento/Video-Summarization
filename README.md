# Video-Summarization

Objective: Given the frames of a 9 minute video, generate a 1.5 minute summary. 
Please see the following presentation slides:<br>
https://docs.google.com/presentation/d/1JzAxssiVaMH7u6PvZ92mhYaOg02fOeh8pYDmhHHBkqg/edit?usp=sharing

 <img src="https://github.com/spesavento/Video-Summarization/blob/main/diagram.png" width="510" height="300.5">

Shot detection was done with SSIM, centering, and color histograms.
 <img src="https://github.com/spesavento/Video-Summarization/blob/main/collage.jpg" width="510" height="400">

Metrics measured were:
- Face Detection
- People Detection
- Block Motion Detection
- Audio

Final video is generated from the resulting frames and the audio is synced. 
