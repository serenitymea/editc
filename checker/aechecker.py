import cv2

class AestheticChecker:
    def __init__(self, loader):
        self.loader = loader

    def group_consecutive(self, frames):
        groups = []
        current = []

        for f in frames:
            if not current or f == current[-1] + 1:
                current.append(f)
            else:
                groups.append(current)
                current = [f]

        if current:
            groups.append(current)

        return groups

    def preview_chunk(self, chunk, delay=40):
        for frame_id in chunk:
            frame = self.loader.get_frames(frame_id)
            if frame is None:
                continue

            cv2.imshow("Aesthetic Checker", frame)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break

    def review(self, frames):
        approved = []
        chunks = self.group_consecutive(frames)

        for chunk in chunks:
            self.preview_chunk(chunk)

            print(f"\nkadry {chunk[0]}–{chunk[-1]}")
            print("Y | N")

            while True:
                key = cv2.waitKey(0)
                if key == ord('y'):
                    approved.extend(chunk)
                    break
                elif key == ord('n'):
                    break

        cv2.destroyAllWindows()
        return approved
