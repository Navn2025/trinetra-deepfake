function getMeetVideos()
{
    const videos=Array.from(
        document.querySelectorAll('video')
    )
    return videos.filter((video) =>
    {
        return (
            video instanceof HTMLVideoElement&&
            video.videoWidth>0 && video.videoHeight>0
        )
    });

}
function getVideoInfo(video) {
    return {
        element: video,
        width: video.videoWidth,
        height: video.videoHeight,
        readyState: video.readyState
    };
}

function logMeetVideos() {
    const videos = getMeetVideos();

    console.log(
        `[MeetVideos] Active videos: ${videos.length}`
    );

    videos.forEach((video, index) => {
        console.log(
            `[MeetVideos] Video ${index}:`,
            getVideoInfo(video)
        );
    });

    return videos;
}