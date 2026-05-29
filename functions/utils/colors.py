import numpy as np

def get_subject_color(subject_name, subject_colors):
    subject_name = str(subject_name)

    if subject_name in subject_colors:
        return np.asarray(subject_colors[subject_name], dtype=float)

    print(f'Warning: no color defined for subject "{subject_name}". Using gray.')
    return np.array([0.4, 0.4, 0.4])


def adjust_color_lightness(base_color, factor):
    base_color = np.asarray(base_color, dtype=float)

    if factor >= 1:
        new_color = 1 - (1 - base_color) / factor
    else:
        new_color = base_color * factor

    return np.clip(new_color, 0, 1)


def make_isi_colors(base_color, n_isi):
    base_color = np.asarray(base_color, dtype=float)

    if n_isi == 1:
        return np.array([base_color])

    factors = np.linspace(1.7, 0.65, n_isi)
    return np.array([adjust_color_lightness(base_color, f) for f in factors])