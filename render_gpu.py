import numpy as np
from glumpy import app, gloo, gl
import cv2
import triangle
import math
import os

# import ukebilde.create_video as ukebilde
# import ukebilde.resize_face as resize_face
# from ukebilde.ukebilde import get_capture_date

morph_vertex_shader = """
    in vec2 position;
    in vec2 a_texcoord;
    out vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(position.x, position.y, 0.0, 1.0);
        v_texcoord = a_texcoord;
    } """

morph_fragment_shader = """
    uniform float weight;
    uniform sampler2D u_texture;
    in vec2 v_texcoord;
    out vec4 fragColor;
    void main() {
        fragColor = texture(u_texture, v_texcoord) * weight;
    } """


blit_vertex_shader = """
    in vec2 position;
    in vec2 a_texcoord;
    out vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(position.x, -position.y, 0.0, 1.0);
        v_texcoord = a_texcoord;
    } """

blit_fragment_shader = """
    uniform sampler2D u_texture;
    in vec2 v_texcoord;
    out vec4 fragColor;
    void main() {
        vec4 tex = texture(u_texture, v_texcoord);
        fragColor = tex.bgra;
    } """


class GpuRenderer(object):
    landmark_count = 72

    def __init__(self, scaled_size):
        self.scaled_size = scaled_size

        #setup render program for rendering a morphing step
        dtype = [
            ("position", np.float32, 2),  # x,y
            ("a_texcoord",    np.float32, 2),  # u,v
        ]
        self.morph_vertices = np.zeros(self.landmark_count, dtype).view(gloo.VertexArray)
        self.morph_indices = np.zeros(1024, dtype=np.uint32).view(gloo.IndexBuffer)
        self.morph_program = gloo.Program(morph_vertex_shader, morph_fragment_shader, version="330")
        self.morph_program.bind(self.morph_indices)
        self.morph_program.bind(self.morph_vertices)

        self.framebuffer_texture = np.zeros((self.scaled_size[1], self.scaled_size[0],4),np.float32).view(gloo.TextureFloat2D)
        self.framebuffer = gloo.FrameBuffer(color=[self.framebuffer_texture])

        #setup render program to blit final morphing result to display buffer
        self.blit_program = gloo.Program(blit_vertex_shader, blit_fragment_shader, count=4, version="330")
        blit_vertices = np.zeros(4, dtype).view(gloo.VertexArray)
        blit_vertices["position"] = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        blit_vertices["a_texcoord"] = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.blit_program.bind(blit_vertices)
        self.blit_program["u_texture"] = self.framebuffer_texture  #input to blit operation is output from the morph

        #setup window, which will 
        config = app.configuration.Configuration()
        config.samples = 4
        config.double_buffer = True
        config.blue_size = 16
        config.red_size = 16
        config.green_size = 16
        config.alpha_size = 0
        config.depth_size = 0
        config.srgb = True
        config.major_version = 4
        config.minor_version = 1
        config.profile = "core"

        w = self.scaled_size[0]
        h = self.scaled_size[1]
        ratio = float(w) / float(h)
        self.window = app.Window(config=config, aspect=ratio, width=w, height=h)
        self.window.push_handlers(on_draw=lambda dt: self._render_frame(dt))

        self.frame_index = 0
        self._prevIndexBuffer = np.array([], dtype=float)

        self.render_to_screen = True

        self.images = None
        self.landmarks = None
        self.weights = None

    def _render_frame(self, dt):
        #create triangularization
        f = triangle.delaunay(self.blended_landmarks).flatten()
        self.morph_indices[:len(self._prevIndexBuffer)] = 0
        self.morph_indices[:len(f)] = f
        self._prevIndexBuffer = f
        self.morph_vertices["position"] = 2 * self.blended_landmarks - 1

        #blend images
        self.framebuffer.activate()
        self.window.clear()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        for img, lm, weight in zip(self.images, self.landmarks, self.weights):
            self.morph_vertices["a_texcoord"] = lm
            self.morph_program["weight"] = weight
            self.morph_program["u_texture"] = img.copy()
            self.morph_program.draw(gl.GL_TRIANGLES, self.morph_indices)
            gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        self.framebuffer.deactivate()
        gl.glDisable(gl.GL_BLEND)

        #convert output to uint8 image
        out_buffer = self.framebuffer.color[0].get()
        self.output_buffer = np.array(out_buffer * 255, dtype=np.uint8)

        if self.render_to_screen:
            #blit to screen framebuffer
            self.blit_program["u_texture"] = self.framebuffer_texture
            self.blit_program.draw(gl.GL_TRIANGLE_STRIP)

    def morph_images(self, images, landmarks, weights, blended_landmarks):
        self.images = images
        self.blended_landmarks = blended_landmarks / np.array(self.scaled_size, dtype=float)
        self.landmarks = landmarks / np.array(self.scaled_size, dtype=float)
        self.weights = weights

        backend  = app.__backend__
        window_count = backend.process(0.1)
        if window_count == 0:
            raise Exception("OpenGL Window Closed")

        return self.output_buffer

