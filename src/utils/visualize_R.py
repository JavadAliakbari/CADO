import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def visualize_high_dim_data(
    X,
    predicted_clusters,  # a list of cluster results
    true_clusters,
    subspaces,  # a list of subspaces arrays (history)
    centers=None,
    filename="3d_plot.html",
    reduction_method="pca",  # new parameter to choose reduction technique
    num_sig_features=2,  # if 2, plot plane instead of lines
    save_path="results/",  # path to save the plot
):
    # Choose dimensionality reduction technique
    if reduction_method.lower() == "tsne":
        from sklearn.manifold import TSNE

        X_3d = TSNE(n_components=3).fit_transform(X)
        # TSNE does not support out-of-sample transform; use PCA as fallback for transforming subspace lines
        from sklearn.decomposition import PCA

        transformer_for_lines = PCA(n_components=3).fit(X)
    elif reduction_method.lower() == "umap":
        import umap

        transformer_for_lines = umap.UMAP(n_components=3)
        X_3d = transformer_for_lines.fit_transform(X)
    elif reduction_method.lower() == "pca":
        from sklearn.decomposition import PCA

        transformer_for_lines = PCA(n_components=3)
        X_3d = transformer_for_lines.fit_transform(X)
    else:
        raise ValueError("Unsupported reduction method")
    # Set fixed axis limits based on X_3d
    x_lim = [float(np.min(X_3d[:, 0])), float(np.max(X_3d[:, 0]))]
    y_lim = [float(np.min(X_3d[:, 1])), float(np.max(X_3d[:, 1]))]
    z_lim = [float(np.min(X_3d[:, 2])), float(np.max(X_3d[:, 2]))]
    # Setup constants
    if centers is None:
        centers = [np.zeros(X.shape[1]) for _ in range(len(subspaces[0]))]
    k = len(centers)
    t_vals = np.linspace(-3, 3, 1000)
    colors = ["red", "green", "blue", "cyan", "magenta", "yellow", "black"]

    # Build initial traces using the first frame data
    init_clusters = predicted_clusters[0]
    init_subspaces = subspaces[0]
    data = []
    for j in range(k):
        idx_correct = (init_clusters == j) & (init_clusters == true_clusters)
        idx_incorrect = (init_clusters == j) & (init_clusters != true_clusters)
        data.append(
            go.Scatter3d(
                x=X_3d[idx_correct, 0],
                y=X_3d[idx_correct, 1],
                z=X_3d[idx_correct, 2],
                mode="markers",
                marker=dict(size=3, color=colors[j % len(colors)], opacity=0.3),
                name=f"Cluster {j}",
            )
        )
        data.append(
            go.Scatter3d(
                x=X_3d[idx_incorrect, 0],
                y=X_3d[idx_incorrect, 1],
                z=X_3d[idx_incorrect, 2],
                mode="markers",
                marker=dict(size=3, color=colors[j % len(colors)], opacity=0.9),
                # name=f"Cluster {j} (incorrect)",
            )
        )
        # Compute local range from points assigned to cluster j
        cluster_points = X_3d[idx_correct, :]
        if cluster_points.shape[0] > 0:
            local_range = np.max(np.linalg.norm(cluster_points, axis=1))
        else:
            local_range = 3  # fallback value

        # Define local t_vals for this cluster
        local_t_vals = np.linspace(-local_range, local_range, 1000)

        if num_sig_features == 2:
            # Plot plane spanned by the last 2 columns
            plane_range = local_range  # use computed local_range
            grid_size = 10
            s_vals = np.linspace(-plane_range, plane_range, grid_size)
            t_vals_plane = np.linspace(-plane_range, plane_range, grid_size)
            S, T = np.meshgrid(s_vals, t_vals_plane)
            # Compute grid of points for the plane:
            grid_points = (
                centers[j]
                + S[..., None] * init_subspaces[j][:, -2]
                + T[..., None] * init_subspaces[j][:, -1]
            )
            grid_points_flat = grid_points.reshape(-1, grid_points.shape[-1])
            grid_points_flat_trans = transformer_for_lines.transform(grid_points_flat)
            grid_points_trans = grid_points_flat_trans.reshape(S.shape + (3,))
            data.append(
                go.Surface(
                    x=grid_points_trans[:, :, 0],
                    y=grid_points_trans[:, :, 1],
                    z=grid_points_trans[:, :, 2],
                    surfacecolor=np.ones_like(grid_points_trans[:, :, 0]),
                    colorscale=[
                        [0, colors[j % len(colors)]],
                        [1, colors[j % len(colors)]],
                    ],
                    opacity=0.3,
                    name=f"Subspace {j} plane",
                    showscale=False,
                )
            )
            # --- NEW: Add a line trace for num_sig_features==2 ---
            # Here we take the last column subspace as the direction for the line.
            line_points = np.array(
                [centers[j] + t * init_subspaces[j][:, -1] for t in local_t_vals]
            )
            line_points_3d = transformer_for_lines.transform(line_points)
            data.append(
                go.Scatter3d(
                    x=line_points_3d[:, 0],
                    y=line_points_3d[:, 1],
                    z=line_points_3d[:, 2],
                    mode="lines",
                    line=dict(color=colors[j % len(colors)], width=4),
                    name=f"Subspace {j} line",
                )
            )
        else:
            for d in range(num_sig_features):
                line_points = np.array(
                    [
                        centers[j] + t * init_subspaces[j][:, -num_sig_features + d]
                        for t in t_vals
                    ]
                )
                line_points_3d = transformer_for_lines.transform(line_points)
                data.append(
                    go.Scatter3d(
                        x=line_points_3d[:, 0],
                        y=line_points_3d[:, 1],
                        z=line_points_3d[:, 2],
                        mode="lines",
                        line=dict(color=colors[j % len(colors)], width=4),
                        name=f"Subspace {j} dir {d}",
                    )
                )
    # Create frames for the animation
    frames = []
    n_frames = len(predicted_clusters)
    for t in range(n_frames):
        frame_data = []
        curr_clusters = predicted_clusters[t]
        curr_subspaces = subspaces[t]
        for j in range(k):
            idx_correct = (curr_clusters == j) & (curr_clusters == true_clusters)
            idx_incorrect = (curr_clusters == j) & (curr_clusters != true_clusters)
            frame_data.append(
                go.Scatter3d(
                    x=X_3d[idx_correct, 0],
                    y=X_3d[idx_correct, 1],
                    z=X_3d[idx_correct, 2],
                    mode="markers",
                    marker=dict(size=3, color=colors[j % len(colors)], opacity=0.3),
                    name=f"Cluster {j}",
                )
            )
            frame_data.append(
                go.Scatter3d(
                    x=X_3d[idx_incorrect, 0],
                    y=X_3d[idx_incorrect, 1],
                    z=X_3d[idx_incorrect, 2],
                    mode="markers",
                    marker=dict(size=3, color=colors[j % len(colors)], opacity=0.9),
                    # name=f"Cluster {j} (incorrect)",
                )
            )
            # Re-compute local range per frame
            cluster_points = X_3d[idx_correct, :]
            if cluster_points.shape[0] > 0:
                local_range = np.max(np.linalg.norm(cluster_points, axis=1))
            else:
                local_range = 3
            local_t_vals = np.linspace(-local_range, local_range, 1000)

            if num_sig_features == 2:
                plane_range = local_range
                grid_size = 10
                s_vals = np.linspace(-plane_range, plane_range, grid_size)
                t_vals_plane = np.linspace(-plane_range, plane_range, grid_size)
                S, T = np.meshgrid(s_vals, t_vals_plane)
                grid_points = (
                    centers[j]
                    + S[..., None] * curr_subspaces[j][:, -2]
                    + T[..., None] * curr_subspaces[j][:, -1]
                )
                grid_points_flat = grid_points.reshape(-1, grid_points.shape[-1])
                grid_points_flat_trans = transformer_for_lines.transform(
                    grid_points_flat
                )
                grid_points_trans = grid_points_flat_trans.reshape(S.shape + (3,))
                frame_data.append(
                    go.Surface(
                        x=grid_points_trans[:, :, 0],
                        y=grid_points_trans[:, :, 1],
                        z=grid_points_trans[:, :, 2],
                        surfacecolor=np.ones_like(grid_points_trans[:, :, 0]),
                        colorscale=[
                            [0, colors[j % len(colors)]],
                            [1, colors[j % len(colors)]],
                        ],
                        opacity=0.5,
                        name=f"Subspace {j} plane",
                        showscale=False,
                    )
                )
                # --- NEW: Add corresponding line trace using local_t_vals.
                line_points = np.array(
                    [centers[j] + t * curr_subspaces[j][:, -1] for t in local_t_vals]
                )
                line_points_3d = transformer_for_lines.transform(line_points)
                frame_data.append(
                    go.Scatter3d(
                        x=line_points_3d[:, 0],
                        y=line_points_3d[:, 1],
                        z=line_points_3d[:, 2],
                        mode="lines",
                        line=dict(color=colors[j % len(colors)], width=4),
                        name=f"Subspace {j} line",
                    )
                )
            else:
                for d in range(num_sig_features):
                    line_points = np.array(
                        [
                            centers[j]
                            + t_val * curr_subspaces[j][:, -num_sig_features + d]
                            for t_val in t_vals
                        ]
                    )
                    line_points_3d = transformer_for_lines.transform(line_points)
                    frame_data.append(
                        go.Scatter3d(
                            x=line_points_3d[:, 0],
                            y=line_points_3d[:, 1],
                            z=line_points_3d[:, 2],
                            mode="lines",
                            line=dict(color=colors[j % len(colors)], width=4),
                            name=f"Subspace {j} dir {d}",
                        )
                    )
        frames.append(
            go.Frame(
                data=frame_data,
                name=str(t),
                layout={
                    "scene": {
                        "xaxis": {"range": x_lim, "autorange": False},
                        "yaxis": {"range": y_lim, "autorange": False},
                        "zaxis": {"range": z_lim, "autorange": False},
                    }
                },
            )
        )
    # Create figure and add slider
    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        title="Clusters and their projected subspaces (animated via dimensionality reduction)",
        scene=dict(
            xaxis=dict(title="PC1", range=x_lim, autorange=False),
            yaxis=dict(title="PC2", range=y_lim, autorange=False),
            zaxis=dict(title="PC3", range=z_lim, autorange=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=-1.25, y=1.25, z=1.25)),
        ),
        uirevision="constant",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": True,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 10, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "Frame: "},
                "pad": {"t": 50},
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [str(t)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": str(t),
                    }
                    for t in range(n_frames)
                ],
            }
        ],
    )
    # Determine indices for plane (surface) traces
    plane_indices = [i for i, trace in enumerate(fig.data) if trace.type == "surface"]
    show_plane = [True] * len(fig.data)
    hide_plane = [True] * len(fig.data)
    for i in plane_indices:
        hide_plane[i] = False

    # Append a new button to toggle plane visibility by updating existing updatemenus
    existing_updatemenus = (
        list(fig.layout.updatemenus) if fig.layout.updatemenus is not None else []
    )
    existing_updatemenus.append(
        {
            "buttons": [
                {
                    "label": "Show Plane",
                    "method": "restyle",
                    "args": [{"visible": show_plane}],
                },
                {
                    "label": "Hide Plane",
                    "method": "restyle",
                    "args": [{"visible": hide_plane}],
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "type": "buttons",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.15,
            "yanchor": "top",
        }
    )
    fig.update_layout(updatemenus=existing_updatemenus)

    # --- ADDITION START ---
    # Identify trace indices for planes, lines, and markers
    plane_indices = [i for i, trace in enumerate(fig.data) if trace.type == "surface"]
    line_indices = [
        i
        for i, trace in enumerate(fig.data)
        if trace.type == "scatter3d" and getattr(trace, "mode", "") == "lines"
    ]
    marker_indices = [
        i for i in range(len(fig.data)) if i not in plane_indices + line_indices
    ]

    # Build visibility arrays for each state:
    # For markers, always True.
    def build_visibility(visible_planes, visible_lines):
        visibility = [True] * len(fig.data)
        for i in plane_indices:
            visibility[i] = visible_planes
        for i in line_indices:
            visibility[i] = visible_lines
        return visibility

    vis_show_plane = build_visibility(True, False)
    vis_show_line = build_visibility(False, True)
    vis_hide = build_visibility(False, False)

    # Append a new updatemenus entry with three buttons:
    existing_updatemenus = (
        list(fig.layout.updatemenus) if fig.layout.updatemenus is not None else []
    )
    existing_updatemenus.append(
        {
            "buttons": [
                {
                    "label": "Show Plane",
                    "method": "restyle",
                    "args": [{"visible": vis_show_plane}],
                },
                {
                    "label": "Show Line",
                    "method": "restyle",
                    "args": [{"visible": vis_show_line}],
                },
                {
                    "label": "Hide",
                    "method": "restyle",
                    "args": [{"visible": vis_hide}],
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "type": "buttons",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.15,
            "yanchor": "top",
        }
    )
    fig.update_layout(updatemenus=existing_updatemenus)
    # --- ADDITION END ---

    fig.write_html(f"{save_path}{filename}")


def visualize_high_dim_data2(
    X,
    predicted_clusters,  # a list of cluster results
    true_clusters,
    subspaces,  # a list of subspaces arrays (history)
    centers=None,
    filename="2d_plot.html",
    reduction_method="tsne",  # new parameter to choose reduction technique
    num_sig_features=2,  # new parameter: number of significant feature directions to plot
    save_path="results/",  # path to save the plot
):
    # Choose dimensionality reduction technique
    if reduction_method.lower() == "tsne":
        from sklearn.manifold import TSNE

        X_2d = TSNE(n_components=2).fit_transform(X)
        # TSNE does not support out-of-sample transform; use PCA as fallback for transforming subspace lines
        from sklearn.decomposition import PCA

        transformer_for_lines = PCA(n_components=2).fit(X)
    elif reduction_method.lower() == "umap":
        import umap

        transformer_for_lines = umap.UMAP(n_components=2)
        X_2d = transformer_for_lines.fit_transform(X)
    elif reduction_method.lower() == "pca":
        from sklearn.decomposition import PCA

        transformer_for_lines = PCA(n_components=2)
        X_2d = transformer_for_lines.fit_transform(X)
    else:
        raise ValueError("Unsupported reduction method")
    # Set fixed axis limits based on X_3d
    x_lim = [float(np.min(X_2d[:, 0])), float(np.max(X_2d[:, 0]))]
    y_lim = [float(np.min(X_2d[:, 1])), float(np.max(X_2d[:, 1]))]
    # z_lim = [float(np.min(X_2d[:, 2])), float(np.max(X_2d[:, 2]))]
    # Setup constants
    if centers is None:
        centers = [np.zeros(X.shape[1]) for _ in range(len(subspaces[0]))]
    k = len(centers)
    t_vals = np.linspace(-3, 3, 1000)
    colors = ["red", "green", "blue", "cyan", "magenta", "yellow", "black"]

    # Build initial traces using the first frame data
    init_clusters = predicted_clusters[0]
    init_subspaces = subspaces[0]
    data = []
    for j in range(k):
        idx_correct = (init_clusters == j) & (init_clusters == true_clusters)
        idx_incorrect = (init_clusters == j) & (init_clusters != true_clusters)
        data.append(
            go.Scatter(
                x=X_2d[idx_correct, 0],
                y=X_2d[idx_correct, 1],
                # z=X_2d[idx_correct, 2],
                mode="markers",
                marker=dict(size=3, color=colors[j % len(colors)], opacity=0.3),
                name=f"Cluster {j}",
            )
        )
        data.append(
            go.Scatter(
                x=X_2d[idx_incorrect, 0],
                y=X_2d[idx_incorrect, 1],
                # z=X_2d[idx_incorrect, 2],
                mode="markers",
                marker=dict(size=3, color=colors[j % len(colors)], opacity=0.9),
                # name=f"Cluster {j} (incorrect)",
            )
        )
        # Subspace line trace for each cluster using transformer_for_lines
        for d in range(num_sig_features):
            line_points = np.array(
                [
                    centers[j] + t * init_subspaces[j][:, -num_sig_features + d]
                    for t in t_vals
                ]
            )
            line_points_2d = transformer_for_lines.transform(line_points)
            data.append(
                go.Scatter(
                    x=line_points_2d[:, 0],
                    y=line_points_2d[:, 1],
                    # z=line_points_2d[:, 2],
                    mode="lines",
                    line=dict(color=colors[j % len(colors)], width=4),
                    name=f"Subspace {j} dir {d}",
                )
            )
    # Create frames for the animation
    frames = []
    n_frames = len(predicted_clusters)
    for t in range(n_frames):
        frame_data = []
        curr_clusters = predicted_clusters[t]
        curr_subspaces = subspaces[t]
        for j in range(k):
            idx_correct = (curr_clusters == j) & (curr_clusters == true_clusters)
            idx_incorrect = (curr_clusters == j) & (curr_clusters != true_clusters)
            frame_data.append(
                go.Scatter(
                    x=X_2d[idx_correct, 0],
                    y=X_2d[idx_correct, 1],
                    # z=X_2d[idx_correct, 2],
                    mode="markers",
                    marker=dict(size=3, color=colors[j % len(colors)], opacity=0.3),
                    name=f"Cluster {j}",
                )
            )
            frame_data.append(
                go.Scatter(
                    x=X_2d[idx_incorrect, 0],
                    y=X_2d[idx_incorrect, 1],
                    # z=X_2d[idx_incorrect, 2],
                    mode="markers",
                    marker=dict(size=3, color=colors[j % len(colors)], opacity=0.9),
                    # name=f"Cluster {j} (incorrect)",
                )
            )
            for d in range(num_sig_features):
                line_points = np.array(
                    [
                        centers[j] + t_val * curr_subspaces[j][:, -num_sig_features + d]
                        for t_val in t_vals
                    ]
                )
                line_points_2d = transformer_for_lines.transform(line_points)
                frame_data.append(
                    go.Scatter(
                        x=line_points_2d[:, 0],
                        y=line_points_2d[:, 1],
                        # z=line_points_2d[:, 2],
                        mode="lines",
                        line=dict(color=colors[j % len(colors)], width=4),
                        name=f"Subspace {j} dir {d}",
                    )
                )
        frames.append(
            go.Frame(
                data=frame_data,
                name=str(t),
                layout={
                    "scene": {
                        "xaxis": {"range": x_lim, "autorange": False},
                        "yaxis": {"range": y_lim, "autorange": False},
                        # "zaxis": {"range": z_lim, "autorange": False},
                    }
                },
            )
        )
    # Create figure and add slider
    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        title="Clusters and their projected subspaces (animated via dimensionality reduction)",
        scene=dict(
            xaxis=dict(title="PC1", range=x_lim, autorange=False),
            yaxis=dict(title="PC2", range=y_lim, autorange=False),
            # zaxis=dict(title="PC3", range=z_lim, autorange=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.25, y=1.25)),
        ),
        uirevision="constant",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 100, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "currentvalue": {"prefix": "Frame: "},
                "pad": {"t": 50},
                "steps": [
                    {
                        "method": "animate",
                        "args": [
                            [str(t)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": str(t),
                    }
                    for t in range(n_frames)
                ],
            }
        ],
    )
    # Save as interactive HTML
    fig.write_html(f"{save_path}{filename}")
