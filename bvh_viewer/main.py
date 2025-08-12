#main.py
import argparse
import math
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import imgui
from imgui.integrations.pygame import PygameRenderer
from pyglm import glm
#import numpy as np

from BVH_Parser import bvh_parser, check_bvh_structure, motion_connect
from Rendering import draw_humanoid, draw_virtual_root_axis
from utils import draw_axes, set_lights, random_color
import Events
import UI

state = {
        'center': glm.vec3(0, 0, 0),
        'eye': glm.vec3(60, 180, 600),
        'upVector': glm.vec3(0, 1, 0),
        'distance': glm.length(glm.vec3(60, 180, 600) - glm.vec3(0, 0, 0)),
        'yaw': math.atan2(60, 600),
        'pitch': math.asin((180) / glm.length(glm.vec3(60, 180, 600))),
        'last_x': 0,
        'last_y': 0,
        'is_rotating': False,
        'is_translating': False,
        'stop': False,
        #'frame_idx': 0,
        #'frame_len': None,
        'root': None,
        'motion_frames': [],
        'motion' : None,
        'loaded_file_path': None,
        'motion_color' : None,
    }

def resize(width, height):
    """
    glViewport사이즈를 조절하는 함수
    :param width: 너비
    :param height: 높이
    """
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, width / height, 0.1, 5000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def main():
    """
    BVH_Viewer 의 main loop
    """
    pygame.init()
    size = (800, 600)
    screen = pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
    pygame.display.set_caption("BVH Viewer with ImGui Control Panel")
    glEnable(GL_DEPTH_TEST)
    set_lights()
    resize(*size)

    imgui.create_context()
    impl = PygameRenderer()

    clock = pygame.time.Clock()
    previous_time = pygame.time.get_ticks() / 1000.0
    frame_duration = 1 / 60.0 #60fps 기준

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            impl.process_event(event)
            io = imgui.get_io()

            if event.type == pygame.MOUSEWHEEL:
                if not io.want_capture_mouse:
                    Events.handle_mouse_wheel(event, state)
            if event.type == pygame.MOUSEMOTION:
                if not io.want_capture_mouse:
                    Events.handle_mouse_motion(event, state)
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                if not io.want_capture_mouse:
                    Events.handle_mouse_button(event, state)
            if event.type == pygame.VIDEORESIZE:
                size = event.size
                screen = pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
                resize(*size)

        io.display_size = pygame.display.get_surface().get_size()
        current_time = pygame.time.get_ticks() / 1000.0
        delta_time = current_time - previous_time
        if not state['stop']:
            if delta_time >= frame_duration and state['motion_frames']:
                state['frame_idx'] = (state['frame_idx'] + 1) % state['frame_len']
                previous_time = current_time

        imgui.new_frame()
        UI.draw_control_panel(state)
        UI.draw_file_loader(state)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(state['eye'].x, state['eye'].y, state['eye'].z,
                  state['center'].x, state['center'].y, state['center'].z,
                  state['upVector'].x, state['upVector'].y, state['upVector'].z)
        draw_axes()

        if state['motion_frames'] and state['root']:
            frame = state['motion_frames'][state['frame_idx']]
            if frame.virtual_transform is None:
                print(f"Frame {state['frame_idx']}: virtual_transform is None!")
            else:
                pos_from_matrix = glm.vec3(frame.virtual_transform[3])
                #if glm.length(pos_from_matrix) < 1e-4 and state['frame_idx'] > 10:  # 처음 몇 프레임 제외
                    # 프레임 인덱스와 위치 정보를 출력
                    #print(f"Frame {state['frame_idx']}: Low translation in virtual_transform -> {pos_from_matrix.x:.2f}, {pos_from_matrix.y:.2f}, {pos_from_matrix.z:.2f}")
            glPushMatrix()
            draw_humanoid(state['root'], frame, state['motion_color'])

            #hip_node = state['root'].children[0]
            virtual_root_t = frame.virtual_transform
            if virtual_root_t is None:
                print("🚨 virtual_root_T is None! Can't draw axis.")
            else:
                draw_virtual_root_axis(virtual_root_t, state['motion_color'])
            glPopMatrix()

        imgui.render()
        impl.render(imgui.get_draw_data())
        pygame.display.flip()
        clock.tick(60)

    impl.shutdown()
    pygame.quit()


# main.py (수정 후)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BVH Viewer and Motion Stitching tool.")
    # nargs='+'는 1개 이상의 파일 경로를 리스트로 받겠다는 의미
    parser.add_argument("file_paths", nargs='+', help="Path to one or two BVH files.")
    parser.add_argument("-t", "--transition", type=int, default=60, help="Number of transition frames for stitching (default: 60)")
    args = parser.parse_args()

    num_files = len(args.file_paths)

    if num_files == 1:
        # --- 단일 파일 뷰어 모드 ---
        print(f"--- Running in Single File Viewer Mode for {args.file_paths[0]} ---")
        
        file_path = args.file_paths[0]
        root, motion = bvh_parser(file_path)
        check_bvh_structure(root, is_root=True)

        # state에 파싱된 모션 정보 등록
        state['root'] = root
        state['motion'] = motion
        state['motion_frames'] = motion.quaternion_frame
        state['frame_len'] = len(motion)
        state['frame_idx'] = 0
        state['motion_color'] = random_color()  # 모션 색상 초기화

    elif num_files == 2:
        # --- 두 파일 스티칭 모드 ---
        print(f"--- Running in Motion Stitching Mode for {args.file_paths[0]} and {args.file_paths[1]} ---")
        
        file_path1 = args.file_paths[0]
        file_path2 = args.file_paths[1]
        
        root1, motion1 = bvh_parser(file_path1)
        root2, motion2 = bvh_parser(file_path2)

        # 골격 구조가 동일한지 확인
        if root1.name != root2.name:
            print(f"Warning: Root joint names are different ('{root1.name}' vs '{root2.name}'). Stitching may produce unexpected results.")
            # 더 정교하게는 전체 구조를 비교해야 함
        
        check_bvh_structure(root1, is_root=True)
        # motion2의 골격 구조는 motion1과 같다고 가정하고 생략 가능
        # check_bvh_structure(root2, is_root=True)

        # 모션 연결 (transition 프레임 수를 인자로 받도록 수정)
        connected_motion = motion_connect(motion1, motion2, root1, transition_frames=args.transition)

        # state에 연결된 모션 등록
        state['root'] = root1  # 기준이 되는 첫 번째 골격 사용
        state['motion'] = connected_motion
        state['motion_frames'] = connected_motion.quaternion_frame
        state['frame_len'] = len(connected_motion)
        state['frame_idx'] = 0
        state['motion_color'] = random_color()

    else:
        # --- 오류 처리 ---
        print("Error: Please provide 1 or 2 BVH file paths.")
        parser.print_help()
        exit(1) # 프로그램 종료

    # main 루프 시작
    main()
