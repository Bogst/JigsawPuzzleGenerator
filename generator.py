from PIL import Image, ImageChops, ImageFilter

from exceptions import CantPlacePieceOnTheBackgroundWithoutOverlap, BackgroundImageNotBigEnough
from utils import get_background_image, get_horizontal_edge_line_points, get_vertical_edge_line_points, \
    draw_puzzle_from_points, transform, save_solution, get_rotation_matrix, get_crop_box_for_piece, \
    draw_solution_on_image

import random
import os


def generate_puzzle(background_images_directory_path: str, puzzle_faces_images_directory_path: str, puzzle_configs) -> ():
    background_image_names = os.listdir(background_images_directory_path)
    input_image_names = os.listdir(puzzle_faces_images_directory_path)

    input_image_path = os.path.join(puzzle_faces_images_directory_path, random.choice(input_image_names))
    input_image = Image.open(input_image_path, mode='r')

    puzzle_configs.fit_on_image(input_image_size=input_image.size)
    if puzzle_configs.LOG_TO_SCREEN:
        print("\rFetching background... ", end="")

    background_finding_attempts = 0
    while True:
        background_image_path = os.path.join(background_images_directory_path, random.choice(background_image_names))
        try:
            background_image = get_background_image(background_image_path, input_image.size)
            break
        except BackgroundImageNotBigEnough as e:
            background_finding_attempts += 1
            if puzzle_configs.LOG_TO_SCREEN:
                print(f"\rBackground image not big enough {e}, trying again, attempt {background_finding_attempts}", end="")
            if background_finding_attempts == 10:
                raise e
    if puzzle_configs.LOG_TO_SCREEN:
        print("\rGenerating puzzle shapes...                    ", end="")

    background_image_with_shapes = Image.new(mode='1', size=background_image.size, color=0)

    horizontal_edge_line_points = get_horizontal_edge_line_points(puzzle_configs)

    vertical_edge_line_points = get_vertical_edge_line_points(horizontal_edge_line_points, puzzle_configs)

    puzzle_outline = draw_puzzle_from_points(vertical_edge_line_points, horizontal_edge_line_points, puzzle_configs)

    # Just the coordinates of notches
    horizontal_notches = []
    vertical_notches = []

    for vertical_line in vertical_edge_line_points:
        horizontal_notches_row = []
        for line_segment in vertical_line:
            x = int(sum([p[0] for p in line_segment[
                                       puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF:-puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF]]) / (
                        len(line_segment[
                            puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF:-puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF])))
            y = int(sum([p[1] for p in line_segment[
                                       puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF:-puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF]]) / (
                        len(line_segment[
                            puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF:-puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF])))
            horizontal_notches_row.append((x, y))
        horizontal_notches.append(horizontal_notches_row)

    for horizontal_line in horizontal_edge_line_points:
        vertical_notches_col = []
        for line_segment in horizontal_line:
            x = int(sum([p[0] for p in line_segment[
                                       puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF:-puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF]]) / (
                        len(line_segment[
                            puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF:-puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF])))
            y = int(sum([p[1] for p in line_segment[
                                       puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF:-puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF]]) / (
                        len(line_segment[
                            puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF:-puzzle_configs.SOLUTION_CONSIDERATION_CUTOFF])))
            vertical_notches_col.append((x, y))
        vertical_notches.append(vertical_notches_col)

    # Formatted coordinates, indexed by row, column, then the two sets of coordinates identifying that connection in the
    # background picture space
    solution = {"vertical_matches": [], "horizontal_matches": []}

    for row_nr in range(puzzle_configs.NUM_ROWS):
        for col_nr in range(puzzle_configs.NUM_COLS):
            if puzzle_configs.LOG_TO_SCREEN:
                print(f"\rCropping and placing piece {row_nr * puzzle_configs.NUM_COLS + col_nr + 1}", end="")

            crop_box = get_crop_box_for_piece(row_nr, col_nr, puzzle_configs)
            crop = puzzle_outline.crop(crop_box)
            piece_shape = Image.new('1', (int(puzzle_configs.COL_WIDTH * 2) + 1, int(puzzle_configs.ROW_HEIGHT * 2) + 1))
            crop_input_image = input_image.crop(crop_box)
            piece = Image.new('RGBA', (int(puzzle_configs.COL_WIDTH * 2) + 1,
                                       int(puzzle_configs.ROW_HEIGHT * 2) + 1),
                              color=(0, 0, 0, 0))

            pixels = crop.getdata()
            p = []
            for row in range(pixels.size[1]):
                p.append([])
                for col in range(pixels.size[0]):
                    p[row].append(pixels[row * pixels.size[0] + col])
            pixels = p
            crop_rows = len(pixels)
            crop_columns = len(pixels[0])

            candidate_x = int(crop_columns / 2)
            candidate_y = int(crop_rows / 2)

            if row_nr == 0:
                candidate_y = 2
            elif row_nr == puzzle_configs.NUM_ROWS - 1:
                candidate_y = crop_rows - 2

            if col_nr == 0:
                candidate_x = 2
            elif col_nr == puzzle_configs.NUM_COLS - 1:
                candidate_x = crop_columns - 2

            #  Flood fill copy
            candidates = [(int(candidate_y), int(candidate_x))]
            offset_x = 0
            offset_y = 0

            if row_nr == 0:
                offset_y += int(puzzle_configs.ROW_HEIGHT * 0.5)
            if col_nr == 0:
                offset_x += int(puzzle_configs.COL_WIDTH * 0.5)
            while len(candidates) > 0:
                curr = candidates.pop()
                if pixels[curr[0]][curr[1]] == 0:
                    if curr[0] > 0:
                        candidates.append((curr[0] - 1, curr[1]))
                    if curr[0] < (crop_rows - 1):
                        candidates.append((curr[0] + 1, curr[1]))
                    if curr[1] > 0:
                        candidates.append((curr[0], curr[1] - 1))
                    if curr[1] < crop_columns - 1:
                        candidates.append((curr[0], curr[1] + 1))
                    pixels[curr[0]][curr[1]] = 1
                    piece_shape.putpixel((curr[1] + offset_x, curr[0] + offset_y), 1)
                    pixel_value = crop_input_image.getpixel((curr[1], curr[0]))
                    # If the image is black and white put the same value on all 3 channels
                    if isinstance(pixel_value, int):
                        pixel_value = [pixel_value, pixel_value, pixel_value]
                    piece.putpixel((curr[1] + offset_x, curr[0] + offset_y),
                                   (pixel_value[0], pixel_value[1], pixel_value[2], 255))

            #  Remove padding spaces for each piece
            piece_bbox = piece.getbbox()
            piece = piece.crop(piece_bbox)
            piece_shape = piece_shape.crop(piece_bbox)

            offset = (offset_x, offset_y)

            def translate(notch, crp, bbox, offst):
                return (notch[0] - crp[0] - bbox[0] + offst[0],
                        notch[1] - crp[1] - bbox[1] + offst[1])

            if col_nr < puzzle_configs.NUM_COLS - 1:
                r_notch = horizontal_notches[col_nr][row_nr]
                new_coord_notch = translate(r_notch, crop_box, piece_bbox, offset)
                if len(solution['horizontal_matches']) <= row_nr:
                    solution['horizontal_matches'].append([])
                solution['horizontal_matches'][row_nr].append([new_coord_notch])

            if col_nr > 0:
                l_notch = horizontal_notches[col_nr - 1][row_nr]
                new_coord_l_notch = translate(l_notch, crop_box, piece_bbox, offset)
                solution['horizontal_matches'][row_nr][col_nr - 1].append(new_coord_l_notch)

            if row_nr < puzzle_configs.NUM_ROWS - 1:
                b_notch = vertical_notches[row_nr][col_nr]
                new_coord_notch = translate(b_notch, crop_box, piece_bbox, offset)
                if len(solution['vertical_matches']) <= row_nr:
                    solution['vertical_matches'].append([])
                solution['vertical_matches'][row_nr].append([new_coord_notch])

            if row_nr > 0:
                t_notch = vertical_notches[row_nr - 1][col_nr]
                new_coord_notch = translate(t_notch, crop_box, piece_bbox, offset)
                solution['vertical_matches'][row_nr - 1][col_nr].append(new_coord_notch)

            #  Placing the piece on the background
            placing_attempt = 0
            while True:
                placing_attempt += 1
                if placing_attempt > 500:
                    raise CantPlacePieceOnTheBackgroundWithoutOverlap
                rotation_angle = random.uniform(0, 360)

                oversize_for_rotation = 300
                oversized_piece = Image.new(mode=piece.mode, size=(piece.size[0] + oversize_for_rotation * 2,
                                                                   piece.size[1] + oversize_for_rotation * 2),
                                            color=0)
                oversized_piece.paste(piece, (oversize_for_rotation, oversize_for_rotation), mask=piece_shape)
                oversized_piece = oversized_piece.rotate(rotation_angle, Image.BICUBIC)

                oversized_piece_shape = Image.new(mode=piece_shape.mode,
                                                  size=(piece_shape.size[0] + oversize_for_rotation * 2,
                                                        piece_shape.size[1] + oversize_for_rotation * 2),
                                                  color=0)
                oversized_piece_shape.paste(piece_shape, (oversize_for_rotation, oversize_for_rotation),
                                            mask=piece_shape)
                oversized_piece_shape = oversized_piece_shape.rotate(rotation_angle, Image.BICUBIC)

                p_bbox = p_shape_bbox = oversized_piece_shape.getbbox()
                p = oversized_piece.crop(p_bbox)
                p_shape = oversized_piece_shape.crop(p_shape_bbox)

                p_shape_width, p_shape_height = p_width, p_height = (p_bbox[2] - p_bbox[0], p_bbox[3] - p_bbox[1])

                piece_position_on_background = (random.randint(0, background_image.size[0] - int(p_width + 10)),
                                                random.randint(0, background_image.size[1] - int(p_height + 10)))

                candidate_position = background_image_with_shapes.crop((piece_position_on_background[0],
                                                                        piece_position_on_background[1],
                                                                        piece_position_on_background[0] + p_shape_width,
                                                                        piece_position_on_background[
                                                                            1] + p_shape_height))

                #  Pieces are not overlapping
                if ImageChops.logical_and(candidate_position, p_shape).getbbox() is None:
                    break

            background_image.paste(p, piece_position_on_background, mask=p_shape)
            background_image_with_shapes.paste(p_shape, piece_position_on_background, mask=p_shape)

            # Mark conection point
            matrix = get_rotation_matrix(oversized_piece.size, rotation_angle)

            def translate(notch, bbox, position_on_background):
                return (notch[0] - bbox[0] + position_on_background[0],
                        notch[1] - bbox[1] + position_on_background[1])

            if col_nr < puzzle_configs.NUM_COLS - 1:
                notch_r = solution['horizontal_matches'][row_nr][col_nr][0]
                notch_r = (transform(notch_r[0] + oversize_for_rotation, notch_r[1] + oversize_for_rotation, matrix))
                notch_r = translate(notch_r, p_bbox, piece_position_on_background)
                solution['horizontal_matches'][row_nr][col_nr][0] = notch_r

            if col_nr > 0:
                notch_l = solution['horizontal_matches'][row_nr][col_nr - 1][-1]
                notch_l = (transform(notch_l[0] + oversize_for_rotation, notch_l[1] + oversize_for_rotation, matrix))
                notch_l = translate(notch_l, p_bbox, piece_position_on_background)
                solution['horizontal_matches'][row_nr][col_nr - 1][-1] = notch_l

            if row_nr < puzzle_configs.NUM_ROWS - 1:
                notch_b = solution['vertical_matches'][row_nr][col_nr][0]
                notch_b = (transform(notch_b[0] + oversize_for_rotation, notch_b[1] + oversize_for_rotation, matrix))
                notch_b = translate(notch_b, p_bbox, piece_position_on_background)
                solution['vertical_matches'][row_nr][col_nr][0] = notch_b

            if row_nr > 0:
                notch_t = solution['vertical_matches'][row_nr - 1][col_nr][-1]
                notch_t = (transform(notch_t[0] + oversize_for_rotation, notch_t[1] + oversize_for_rotation, matrix))
                notch_t = translate(notch_t, p_bbox, piece_position_on_background)
                solution['vertical_matches'][row_nr - 1][col_nr][-1] = notch_t

    if puzzle_configs.GAUSSIAN_BLUR:
        background_image = background_image.filter(ImageFilter.GaussianBlur(radius=puzzle_configs.GAUSSIAN_BLUR_RADIUS))

    if puzzle_configs.SHOW_SOLUTION:
        draw_solution_on_image(background_image_with_shapes, solution, puzzle_configs)
    return background_image, solution, background_image_with_shapes, puzzle_outline


def generate_puzzles(nr_puzzles, path, save_path, puzzle_configs):
    idx = 0
    for _ in range(nr_puzzles):
        input = None
        while input is None:
            try:
                input, solution, shapes, _ = generate_puzzle(os.path.join(path, 'background'),
                                                             os.path.join(path, 'puzzle_faces'),
                                                             puzzle_configs=puzzle_configs)
            except (BackgroundImageNotBigEnough, CantPlacePieceOnTheBackgroundWithoutOverlap):
                continue

        existing_files = os.listdir(save_path)
        existing_files = [int(file.split('.')[0]) for file in existing_files if 'shape' not in file]
        if len(existing_files) == 0:
            last_puzzle = -1
        else:
            last_puzzle = max(existing_files)
        input.save(os.path.join(save_path, f"{last_puzzle+1}.jpg"))
        shapes.save(os.path.join(save_path, f"{last_puzzle+1}_shape.jpg"))
        save_solution(solution, os.path.join(save_path, f"{last_puzzle + 1}.json"))
        idx += 1
        if puzzle_configs.LOG_TO_SCREEN:
            print(f'\r Idx: {idx}', end="")



