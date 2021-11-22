# Helper methods
def get_encoded_points_and_labels(model, image_set, num_points=None):
    points = []
    labels = []
    if num_points is None:
        num_points = len(image_set)
    for i in range(num_points):
        img = image_set[i][0].unsqueeze(0)
        label = image_set[i][1]
        labels.append(label)
        with torch.no_grad():
            encoded_img = model.encode(img)
            rec_img = model.decode(encoded_img)
            encoded_img = encoded_img.flatten().numpy()
        points.append(encoded_img)
    return points, labels


def make_simplicial_complex(points, diameter):
    skeleton = gd.RipsComplex(points=points, max_edge_length=diameter)
    simplex_tree = skeleton.create_simplex_tree(max_dimension=2)
    return simplex_tree


def get_persistence_features(simplex_tree):
    simplex_tree.persistence()
    # barcodes = simplex_tree.persistence()
    zero_dim_features = simplex_tree.persistence_intervals_in_dimension(0)
    one_dim_features = simplex_tree.persistence_intervals_in_dimension(1)
    two_dim_features = simplex_tree.persistence_intervals_in_dimension(2)

    features = [zero_dim_features, one_dim_features, two_dim_features]
    # features = normalize_features(features, 1000)
    return features


def normalize_features(features, max_value):
    for k_dim_features in features:
        for feature in k_dim_features:
            if feature[1] == float('inf'):
                feature[1] = max_value
    return features


def get_persistence_landscapes(features, num_landscapes, diameter, resolution=100):
    landscape = gd.representations.Landscape(num_landscapes=num_landscapes, 
                                             resolution=resolution,
                                             sample_range=[0, diameter])
    landscape_vectors = np.zeros((len(features), num_landscapes * resolution))
    for i, k_dim_features in enumerate(features):
        if len(k_dim_features) > 0:
            landscape_vector = landscape.fit_transform([k_dim_features])
            landscape_vectors[i] += landscape_vector.flatten()
    return np.array(landscape_vectors)