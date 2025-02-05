import imageio
from vispy import app, gloo

def main():
    # Open webcam
    reader = imageio.get_reader(0)  # Use '0' for default webcam


    # Create a Vispy window
    canvas = app.Canvas(keys='interactive', size=(640, 480))
    texture = gloo.Texture2D((480, 640, 3))

    @canvas.connect
    def on_draw(event):
        frame = reader.get_next_data()
        texture.set_data(frame)
        gloo.clear()
        gloo.blit(texture, (0, 0), canvas.size)

    canvas.show()
    app.run()

if __name__ == "__main__":
    main()
