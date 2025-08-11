import imgui
from tkinter import filedialog
from BVH_Parser import bvh_parser
def draw_control_panel(state):
    """
    Animation의 재생/멈춤 그리고 slider 바를 표시하는 Control Panel입니다.
    :param state: Animation 정보를 담고있는 딕셔너리
    """
    imgui.set_next_window_position(10, 10)
    imgui.begin("Control Panel")
    changed, value = imgui.slider_int("Slider", state['frame_idx']+1, 0, state['frame_len'])
    if imgui.button("play/pause", 100, 30):
        state['stop'] = not state['stop']
    if changed:
        state['frame_idx'] = value-1
    imgui.separator()
    imgui.text("Joint Tree:")
    if state['root']:
        draw_joint_tree(state['root'])
    imgui.end()

def draw_joint_tree(joint):
    """
    Skeleton의 Hierarchy를 표시하는 창입니다.

    :param joint: 계층별로 존재하는 Joint를 그리기 위함입니다.
    """
    node_open = imgui.tree_node(joint.name)
    if imgui.is_item_clicked():
        print("Joint selected:", joint.name)
    if node_open:
        for child in joint.children:
            draw_joint_tree(child)
        imgui.tree_pop()

def draw_file_loader(state):
    """
    BVH_Data를 폴더 탐색기에서 직접 I/O 하기 위한 창입니다.
    :param state: 현재 선택된 파일 경로를 담고 있는 state 딕셔너리
    """
    imgui.set_next_window_position(550, 10, condition=imgui.ONCE)
    imgui.set_next_window_size(200, 100, condition=imgui.ONCE)
    imgui.begin("BVH Loader")
    if imgui.button("Open Folder/Directory"):
        file_path = filedialog.askopenfilename(
            title="Select BVH file",
            filetypes=[("BVH Files", "*.bvh")]
        )
        if file_path:
            state['loaded_file_path'] = file_path
            root, motion_frames = bvh_parser(file_path)
            state['root'] = root
            state['motion_frames'] = motion_frames
            print("File selected:", file_path)
    if state.get('loaded_file_path'):
        imgui.text("Loaded: {}".format(state['loaded_file_path'].split("/")[-1]))
    imgui.end()
