import matplotlib.pyplot as plt


def get_tab10_color(index, cycle=True):
    """
    Generates a color from the Matplotlib 'tab10' colormap.

    The 'tab10' colormap provides 10 distinct colors.

    Args:
        index (int): The integer index of the desired color (0 to 9).
        cycle (bool): If True, the index will wrap around if it exceeds 9.
                      If False and 'index' is out of bounds (not 0-9), a ValueError is raised.

    Returns:
        tuple: An RGBA tuple (red, green, blue, alpha) where each component
               is between 0 and 1.

    Raises:
        ValueError: If 'cycle' is False and 'index' is outside the 0-9 range.
    """
    if not isinstance(index, int):
        raise TypeError("Index must be an integer for 'tab10' colormap.")

    num_colors = 10
    if cycle:
        effective_index = index % num_colors
    else:
        if not (0 <= index < num_colors):
            raise ValueError(
                f"Index {index} is out of bounds for 'tab10' colormap which has "
                f"{num_colors} colors. Set 'cycle=True' to cycle colors."
            )
        effective_index = index

    # Matplotlib's qualitative colormaps can often be indexed directly
    # by normalizing to the 0-1 range for the cmap callable.
    # We sample at the center of each 'bin' for discrete colors.
    normalized_value = (effective_index + 0.5) / num_colors

    # Get the RGBA color where each component is between 0 and 1
    rgba_float = plt.colormaps["tab10"](normalized_value)

    # Convert RGBA (0-1 floats) to RGB (0-255 integers)
    # We ignore the alpha channel as RGB (0-255) typically doesn't include it.
    rgb_255 = tuple(int(c * 255) for c in rgba_float[:3])

    return rgb_255
