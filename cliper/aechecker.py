import cv2

class AestheticChecker:
    def __init__(self, loader):
        self.loader = loader

    def preview_clip(self, clip, delay=40):
        for i, frame in self.loader.frames(clip.start_frame, clip.end_frame + 1):
            cv2.imshow("Aesthetic Checker", frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                return False
        return True

    def review(self, clips):
        approved = []

        for clip in clips:
            shown = self.preview_clip(clip)

            if not shown:
                break

            print(
                f"\nClip {clip.start_frame}-{clip.end_frame} "
                f"({clip.start_time:.2f}s – {clip.end_time:.2f}s, "
                f"{clip.duration:.2f}s, score={clip.score:.2f})"
            )
            print("Y | N | F")

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('y'):
                    approved.append(clip)
                    break
                if key == ord('n'):
                    break
                if key == ord('f'):
                    cv2.destroyAllWindows()
                    return approved

        cv2.destroyAllWindows()
        return approved
        
