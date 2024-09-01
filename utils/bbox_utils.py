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

def measure_distance(point1, point2):
    """Measure the Euclidean distance between two points.

    Args:
        point1 (tuple): A tuple of two integers representing the first point.
        point2 (tuple): A tuple of two integers representing the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]

def get_foot_position(player_bbox):
    """Get the foot position of a player.

    Args:
        player_bbox (list): A list of four integers representing the player's bounding box.

    Returns:
        tuple: A tuple of two integers representing the foot position of the player.
    """
    x1, y1, x2, y2 = player_bbox
    return int((x1 + x2) / 2), int(y2)
