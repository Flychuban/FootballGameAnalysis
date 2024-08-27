def get_center_of_bbox(bbox):
    """Get the center of a bounding box.

    Args:
        bbox (list): A list of four integers representing the bounding box.

    Returns:
        tuple: A tuple of two integers representing the center of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox):
    """Get the width of a bounding box.

    Args:
        bbox (list): A list of four integers representing the bounding box.

    Returns:
        int: The width of the bounding box.
    """
    x1, _, x2, _ = bbox
    return abs(x2 - x1)