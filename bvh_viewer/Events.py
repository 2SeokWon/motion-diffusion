import math
from pygame.locals import *
import pygame
from pyglm import glm

"""
마우스, 키보드 event를 처리하기 위한 함수입니다.
"""

def update_eye(center, distance, yaw, pitch):
    """
    변화하는 input값에 따라 카메라를 update하는 함수입니다
    :param center: 카메라 중심값
    :param distance: 카메라 거리값
    :param yaw: 카메라 yaw 값 (rotation)
    :param pitch: 카메라 pitch 값 (rotation)
    """
    max_pitch = math.radians(89)
    pitch = max(-max_pitch, min(pitch, max_pitch))
    offset_x = distance * math.sin(yaw) * math.cos(pitch)
    offset_y = distance * math.sin(pitch)
    offset_z = distance * math.cos(yaw) * math.cos(pitch)
    return center + glm.vec3(offset_x, offset_y, offset_z)

def handle_mouse_motion(event, state):
    """
    마우스 입력을 처리하는 함수입니다.
    :param event: 이벤트 (드래그)를 분기점으로 하기 위한 param
    :param state: main에서 프로그램을 관리하기 위한 global 한 딕셔너리
    """
    xpos, ypos = event.pos
    dx = xpos - state['last_x']
    dy = ypos - state['last_y']
    state['last_x'], state['last_y'] = xpos, ypos

    if state['is_rotating']:
        sensitivity = 0.005
        state['yaw'] -= dx * sensitivity
        state['pitch'] += dy * sensitivity
        state['eye'] = update_eye(state['center'], state['distance'], state['yaw'], state['pitch'])
    elif state['is_translating']:
        sensitivity = 0.005 * state['distance']
        view_dir = glm.normalize(state['center'] - state['eye'])
        right = glm.normalize(glm.cross(view_dir, glm.vec3(0, 1, 0)))
        up = glm.vec3(0, 1, 0)
        translation = (-dx * right + dy * up) * sensitivity
        state['center'] += translation
        state['eye'] = update_eye(state['center'], state['distance'], state['yaw'], state['pitch'])

def handle_mouse_button(event, state):
    """
    마우스 클릭을 처리하는 함수입니다.
    :param event: 이벤트 (클릭)를 분기점으로 하기 위한 param
    :param state: main에서 프로그램을 관리하기 위한 global 한 딕셔너리
    """
    if event.type == pygame.MOUSEBUTTONDOWN:
        state['last_x'], state['last_y'] = event.pos
        if event.button == 1:
            state['is_rotating'] = True
        elif event.button == 3:
            state['is_translating'] = True
        if event.button in (4, 5):
            handle_mouse_wheel(event, state)
    elif event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
            state['is_rotating'] = False
        elif event.button == 3:
            state['is_translating'] = False

def handle_mouse_wheel(event, state):
    """
    마우스 스크롤을 처리하는 함수입니다.
    :param event: 이벤트 (스크롤)를 분기점으로 하기 위한 param
    :param state: main에서 프로그램을 관리하기 위한 global 한 딕셔너리
    """
    zoom_sensitivity = 0.1
    if hasattr(event, 'y'):
        state['distance'] -= zoom_sensitivity * event.y * state['distance']
    else:
        if event.button == 4:
            state['distance'] -= zoom_sensitivity * state['distance']
        elif event.button == 5:
            state['distance'] += zoom_sensitivity * state['distance']
    state['distance'] = max(state['distance'], 0.1)
    state['eye'] = update_eye(state['center'], state['distance'], state['yaw'], state['pitch'])
