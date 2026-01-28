import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

colors = ["red", "blue", "green", "orange", "cyan", "magenta", "yellow", "black"]


def visualize_kmeans_clustering_2d(
    X,
    predicted_clusters,  # a list of cluster results over iterations
    true_clusters,
    cluster_centers,  # a list of cluster centers arrays (history) - shape (iters, k, d)
    tau,  # maximum norm constraint for cluster centers
    filename="kmeans_2d_plot.html",
    reduction_method="pca",  # 'pca', 'tsne', or 'umap'
    save_path="results/",
):
    """
    Visualize k-means clustering results in 2D with animated cluster evolution.
    Shows cluster centers with constraint circles of radius tau.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data points
    predicted_clusters : list of arrays
        List of predicted cluster assignments over iterations
    true_clusters : array-like, shape (n_samples,)
        Ground truth cluster labels
    cluster_centers : list of arrays
        List of cluster centers (theta values) over iterations, shape (k, d) per iteration
    tau : float
        Maximum norm constraint for cluster centers (||theta|| <= tau)
    filename : str
        Output HTML filename
    reduction_method : str
        Dimensionality reduction method: 'pca', 'tsne', or 'umap'
    save_path : str
        Directory to save the visualization
    """
    X_2d = X

    # Set fixed axis limits based on X_2d with some padding
    x_range = X_2d[:, 0].max() - X_2d[:, 0].min()
    y_range = X_2d[:, 1].max() - X_2d[:, 1].min()
    padding = 0.15 * max(x_range, y_range)

    x_lim = [float(X_2d[:, 0].min() - padding), float(X_2d[:, 0].max() + padding)]
    y_lim = [float(X_2d[:, 1].min() - padding), float(X_2d[:, 1].max() + padding)]

    # Ensure both axes have the same range to maintain aspect ratio
    max_range = max(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0])
    x_center = (x_lim[0] + x_lim[1]) / 2
    y_center = (y_lim[0] + y_lim[1]) / 2
    x_lim = [x_center - max_range / 2, x_center + max_range / 2]
    y_lim = [y_center - max_range / 2, y_center + max_range / 2]

    k = len(cluster_centers[0])  # Number of clusters

    # Build initial traces using the first frame data
    init_clusters = predicted_clusters[0]
    init_centers = cluster_centers[0]

    data = []

    # Plot data points for each cluster
    for j in range(k):
        # Correctly predicted points
        idx_correct = (init_clusters == j) & (init_clusters == true_clusters)
        # Incorrectly predicted points
        idx_incorrect = (init_clusters == j) & (init_clusters != true_clusters)

        # Correct predictions (more transparent)
        data.append(
            go.Scatter(
                x=X_2d[idx_correct, 0],
                y=X_2d[idx_correct, 1],
                mode="markers",
                marker=dict(size=6, color=colors[j % len(colors)], opacity=0.5),
                name=f"Cluster {j}",
            )
        )

        # Incorrect predictions (more opaque with 'x' symbol)
        data.append(
            go.Scatter(
                x=X_2d[idx_incorrect, 0],
                y=X_2d[idx_incorrect, 1],
                mode="markers",
                marker=dict(
                    size=8,
                    color=colors[j % len(colors)],
                    opacity=0.9,
                    symbol="x",
                ),
                showlegend=False,
            )
        )

    # Plot cluster centers and constraint circles
    centers_2d = init_centers

    for j in range(k):
        # Add cluster center
        data.append(
            go.Scatter(
                x=[centers_2d[j, 0]],
                y=[centers_2d[j, 1]],
                mode="markers",
                marker=dict(
                    size=15,
                    color=colors[j % len(colors)],
                    symbol="diamond",
                    line=dict(color="black", width=2),
                ),
                name=f"Center {j}",
                showlegend=(j == 0),  # Only show legend for first center
            )
        )

        # Add constraint circle around center with radius tau (in transformed space)
        # We need to transform a circle of radius tau in original space
        # Create points on a circle in the original d-dimensional space
        theta_angles = np.linspace(0, 2 * np.pi, 100)

        # For 2D visualization, we project the constraint sphere onto the PC plane
        # The radius in PC space is approximately tau scaled by the variance explained
        # if hasattr(pca_for_centers, "explained_variance_ratio_"):
        #     # For PCA, scale by sqrt of variance
        #     radius_scale = np.sqrt(pca_for_centers.explained_variance_ratio_[:2].sum())
        # else:
        radius_scale = 1.0  # For TSNE/UMAP, use approximate scaling

        circle_x = centers_2d[j, 0] + tau * radius_scale * np.cos(theta_angles)
        circle_y = centers_2d[j, 1] + tau * radius_scale * np.sin(theta_angles)

        data.append(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode="lines",
                line=dict(color=colors[j % len(colors)], width=2, dash="dash"),
                name=f"τ={tau}" if j == 0 else None,
                showlegend=(j == 0),
            )
        )

    # Create frames for animation
    frames = []
    for frame_idx, (pred_clust, centers) in enumerate(
        zip(predicted_clusters, cluster_centers)
    ):
        frame_data = []

        # Update data points for each cluster
        for j in range(k):
            idx_correct = (pred_clust == j) & (pred_clust == true_clusters)
            idx_incorrect = (pred_clust == j) & (pred_clust != true_clusters)

            frame_data.append(
                go.Scatter(
                    x=X_2d[idx_correct, 0],
                    y=X_2d[idx_correct, 1],
                    mode="markers",
                    marker=dict(size=6, color=colors[j % len(colors)], opacity=0.5),
                )
            )

            frame_data.append(
                go.Scatter(
                    x=X_2d[idx_incorrect, 0],
                    y=X_2d[idx_incorrect, 1],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=colors[j % len(colors)],
                        opacity=0.9,
                        symbol="x",
                    ),
                )
            )

        # Update cluster centers and circles
        centers_2d = centers

        for j in range(k):
            # Center point
            frame_data.append(
                go.Scatter(
                    x=[centers_2d[j, 0]],
                    y=[centers_2d[j, 1]],
                    mode="markers",
                    marker=dict(
                        size=15,
                        color=colors[j % len(colors)],
                        symbol="diamond",
                        line=dict(color="black", width=2),
                    ),
                )
            )

            # Constraint circle
            theta_angles = np.linspace(0, 2 * np.pi, 100)
            # if hasattr(pca_for_centers, "explained_variance_ratio_"):
            #     radius_scale = np.sqrt(
            #         pca_for_centers.explained_variance_ratio_[:2].sum()
            #     )
            # else:
            radius_scale = 1.0

            circle_x = centers_2d[j, 0] + tau * radius_scale * np.cos(theta_angles)
            circle_y = centers_2d[j, 1] + tau * radius_scale * np.sin(theta_angles)

            frame_data.append(
                go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode="lines",
                    line=dict(color=colors[j % len(colors)], width=2, dash="dash"),
                )
            )

        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))

    # Create the figure
    fig = go.Figure(data=data, frames=frames)

    # Update layout with animation controls
    fig.update_layout(
        title="K-Means Clustering Evolution (2D)",
        xaxis=dict(
            range=x_lim,
            title="PC1" if reduction_method == "pca" else "Dim 1",
            constrain="domain",
        ),
        yaxis=dict(
            range=y_lim,
            title="PC2" if reduction_method == "pca" else "Dim 2",
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        autosize=False,
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "y": 0.05,
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Iteration: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "steps": [
                    {
                        "args": [
                            [f],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "method": "animate",
                        "label": str(f),
                    }
                    for f in range(len(frames))
                ],
            }
        ],
        height=700,
        width=900,
    )

    # Save the figure
    fig.write_html(f"{save_path}{filename}")
    print(f"K-means 2D visualization saved to {save_path}{filename}")


def visualize_kmeans_clustering(
    X,
    predicted_clusters,  # a list of cluster results over iterations
    true_clusters,
    cluster_centers,  # a list of cluster centers arrays (history) - shape (iters, k, d)
    filename="3d_plot.html",
    reduction_method="pca",  # 'pca', 'tsne', or 'umap'
    save_path="results/",
):
    """
    Visualize k-means clustering results in 3D with animated cluster evolution.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        The input data points
    predicted_clusters : list of arrays
        List of predicted cluster assignments over iterations
    true_clusters : array-like, shape (n_samples,)
        Ground truth cluster labels
    cluster_centers : list of arrays
        List of cluster centers (theta values) over iterations, shape (k, d) per iteration
    filename : str
        Output HTML filename
    reduction_method : str
        Dimensionality reduction method: 'pca', 'tsne', or 'umap'
    save_path : str
        Directory to save the visualization
    """
    # Choose dimensionality reduction technique
    if X.shape[1] == 3:
        X_3d = X
        # pca_for_centers = PCA(n_components=3)
        # pca_for_centers.fit(X)
    elif reduction_method.lower() == "tsne":
        from sklearn.manifold import TSNE

        transformer = TSNE(n_components=3, random_state=42)
        X_3d = transformer.fit_transform(X)
        # For cluster centers, use PCA as TSNE doesn't support transform
        pca_for_centers = PCA(n_components=3)
        pca_for_centers.fit(X)
    elif reduction_method.lower() == "pca":
        transformer = PCA(n_components=3)
        X_3d = transformer.fit_transform(X)
        pca_for_centers = transformer
    else:
        raise ValueError(f"Unsupported reduction method: {reduction_method}")

    # Set fixed axis limits based on X_3d
    x_lim = [float(np.min(X_3d[:, 0])), float(np.max(X_3d[:, 0]))]
    y_lim = [float(np.min(X_3d[:, 1])), float(np.max(X_3d[:, 1]))]
    z_lim = [float(np.min(X_3d[:, 2])), float(np.max(X_3d[:, 2]))]

    k = len(cluster_centers[0])  # Number of clusters

    # Build initial traces using the first frame data
    init_clusters = predicted_clusters[0]
    init_centers = cluster_centers[0]

    data = []

    # Plot data points for each cluster
    for j in range(k):
        # Correctly predicted points
        idx_correct = (init_clusters == j) & (init_clusters == true_clusters)
        # Incorrectly predicted points
        idx_incorrect = (init_clusters == j) & (init_clusters != true_clusters)

        # Correct predictions (more transparent)
        data.append(
            go.Scatter3d(
                x=X_3d[idx_correct, 0],
                y=X_3d[idx_correct, 1],
                z=X_3d[idx_correct, 2],
                mode="markers",
                marker=dict(size=4, color=colors[j % len(colors)], opacity=0.4),
                name=f"Cluster {j}",
            )
        )

        # Incorrect predictions (more opaque to highlight errors)
        data.append(
            go.Scatter3d(
                x=X_3d[idx_incorrect, 0],
                y=X_3d[idx_incorrect, 1],
                z=X_3d[idx_incorrect, 2],
                mode="markers",
                marker=dict(
                    size=4,
                    color=colors[j % len(colors)],
                    opacity=0.9,
                    symbol="x",  # Use 'x' symbol for incorrect predictions
                ),
                showlegend=False,
            )
        )

    # Plot cluster centers
    if X.shape[1] == 3:
        centers_3d = init_centers
    else:
        centers_3d = pca_for_centers.transform(init_centers)
    data.append(
        go.Scatter3d(
            x=centers_3d[:, 0],
            y=centers_3d[:, 1],
            z=centers_3d[:, 2],
            mode="markers",
            marker=dict(
                size=12,
                color=[colors[j % len(colors)] for j in range(k)],
                symbol="diamond",
                line=dict(color="black", width=2),
            ),
            name="Centers",
        )
    )

    # Create frames for animation
    frames = []
    for frame_idx, (pred_clust, centers) in enumerate(
        zip(predicted_clusters, cluster_centers)
    ):
        frame_data = []

        # Update data points for each cluster
        for j in range(k):
            idx_correct = (pred_clust == j) & (pred_clust == true_clusters)
            idx_incorrect = (pred_clust == j) & (pred_clust != true_clusters)

            frame_data.append(
                go.Scatter3d(
                    x=X_3d[idx_correct, 0],
                    y=X_3d[idx_correct, 1],
                    z=X_3d[idx_correct, 2],
                    mode="markers",
                    marker=dict(size=4, color=colors[j % len(colors)], opacity=0.4),
                )
            )

            frame_data.append(
                go.Scatter3d(
                    x=X_3d[idx_incorrect, 0],
                    y=X_3d[idx_incorrect, 1],
                    z=X_3d[idx_incorrect, 2],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=colors[j % len(colors)],
                        opacity=0.9,
                        symbol="x",
                    ),
                )
            )

        # Update cluster centers
        if X.shape[1] == 3:
            centers_3d = centers
        else:
            centers_3d = pca_for_centers.transform(centers)
        frame_data.append(
            go.Scatter3d(
                x=centers_3d[:, 0],
                y=centers_3d[:, 1],
                z=centers_3d[:, 2],
                mode="markers",
                marker=dict(
                    size=12,
                    color=[colors[j % len(colors)] for j in range(k)],
                    symbol="diamond",
                    line=dict(color="black", width=2),
                ),
            )
        )

        frames.append(go.Frame(data=frame_data, name=str(frame_idx)))

    # Create the figure
    fig = go.Figure(data=data, frames=frames)

    # Update layout with animation controls
    fig.update_layout(
        title="K-Means Clustering Evolution",
        scene=dict(
            xaxis=dict(
                range=x_lim, title="PC1" if reduction_method == "pca" else "Dim 1"
            ),
            yaxis=dict(
                range=y_lim, title="PC2" if reduction_method == "pca" else "Dim 2"
            ),
            zaxis=dict(
                range=z_lim, title="PC3" if reduction_method == "pca" else "Dim 3"
            ),
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "y": 0.05,
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Iteration: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "steps": [
                    {
                        "args": [
                            [f],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "method": "animate",
                        "label": str(f),
                    }
                    for f in range(len(frames))
                ],
            }
        ],
    )

    # Save the figure
    fig.write_html(f"{save_path}{filename}")
    print(f"K-means visualization saved to {save_path}{filename}")


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

        transformer_for_lines = PCA(n_components=3).fit(X)
    elif reduction_method.lower() == "pca":

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
