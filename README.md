# Video-Summarization

Objective: Given the frames of any 9 minute video, generate a 1.5 minute summary. The summary is made up of the shots (no camera cut) with the highest weighting based on on chosen measures of face detection, people detection, block motion detection, and audio analysis.
<br>
<br>
Please see the following presentation slides:<br>
https://docs.google.com/presentation/d/1JzAxssiVaMH7u6PvZ92mhYaOg02fOeh8pYDmhHHBkqg/edit?usp=sharing

 <img src="https://github.com/spesavento/Video-Summarization/blob/main/diagram.png" width="670" height="300.5">

Shot detection is done with SSIM, centering, and color histograms. <br> 
<br>
 <img src="https://github.com/spesavento/Video-Summarization/blob/main/collage.jpg" width="510" height="400">

Metrics measured are:
- Face Detection
- People Detection
- Block Motion Detection
- Audio

The final video is generated from synching the audio with the resulting summary frames. 
