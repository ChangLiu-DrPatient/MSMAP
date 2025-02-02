import random, argparse, pickle, ast, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, multivariate_normal

cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)

np.random.seed(30)
random.seed(30)


def generate_mass_spectrum(start=0, end=1000, resolution=1, peaks=[[100, 1, 1]], noise_std=1):
    """
    Generate a mass spectrum with specified peaks and global Gaussian noise.

    Parameters:
        start (float): The starting value of the x-axis (e.g., m/z).
        end (float): The ending value of the x-axis (e.g., m/z).
        resolution (float): The resolution or step size for the x-axis.
        peaks (list of lists): A list of lists where each contains:
            - location (float): The x-coordinate of the peak.
            - intensity (float): The height of the peak.
            - width (float): The standard deviation of the Gaussian peak.
        noise_std (float, optional): The standard deviation of the global Gaussian noise. Default is 0.1.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): The x-axis values.
            - y (numpy.ndarray): The y-axis values with peaks and noise.
    """
    # Generate the x-axis values
    x = np.arange(start, end, resolution)

    # Initialize the y-axis values with zeros
    y = np.zeros_like(x)

    # Add peaks to the spectrum
    for location, intensity, width in peaks:
        y += intensity * np.exp(-0.5 * ((x - location) / width) ** 2)

    # Add global Gaussian noise
    noise = np.abs(np.random.normal(0, noise_std, size=x.shape))
    y += noise

    return x, y


def sample_points_in_circle(center_x, center_y, radius, num_points):
    """
    Sample random points uniformly from within a 2D circle.

    Parameters:
        center_x (float): x-coordinate of the circle's center.
        center_y (float): y-coordinate of the circle's center.
        radius (float): Radius of the circle.
        num_points (int): Number of points to sample.

    Returns:
        numpy.ndarray: An array of shape (num_points, 2) containing the sampled points.
    """
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = np.sqrt(np.random.uniform(0, 1, num_points)) * radius
    x = center_x + radii * np.cos(angles)
    y = center_y + radii * np.sin(angles)
    return np.column_stack((x, y)).tolist()


def sample_points_from_gaussian(center_x, center_y, std, num_points, bounds_x=(0,1), bounds_y=(0,1)):
    """
    Sample random points from a 2D truncated normal distribution within given bounds.

    Parameters:
        center_x (float): x-coordinate of the mean of the Gaussian.
        center_y (float): y-coordinate of the mean of the Gaussian.
        std (float): Standard deviation of the Gaussian (same for both dimensions).
        num_points (int): Number of points to sample.
        bounds_x (tuple): Tuple (min_x, max_x) defining the bounds for the x-coordinate.
        bounds_y (tuple): Tuple (min_y, max_y) defining the bounds for the y-coordinate.

    Returns:
        list: A list of sampled points as (x, y) tuples.
    """
    mean = np.array([center_x, center_y])
    cov = np.array([[std**2, 0], [0, std**2]])  # Covariance matrix for the Gaussian
    points = []

    while len(points) < num_points:
        # Sample from the multivariate normal distribution
        samples = multivariate_normal.rvs(mean=mean, cov=cov, size=num_points)

        # Filter points within bounds
        valid_points = [
            tuple(sample) for sample in samples
            if bounds_x[0] <= sample[0] <= bounds_x[1] and bounds_y[0] <= sample[1] <= bounds_y[1]
        ]

        points.extend(valid_points[:num_points - len(points)])  # Add valid points until num_points is reached

    return points


def generate_circles_with_constraints(num_circles, min_radius, max_radius, threshold):
    """
    Generate circles within the square [0, 1] x [0, 1] such that:
        - All circles have diameter < threshold.
        - The distance between the centers of any two circles with radii R and r is at least R + r + threshold.

    Parameters:
        num_circles (int): Number of circles to generate.
        min_radius (float): Minimum radius of the circles.
        max_radius (float): Maximum radius of the circles.
        threshold (float): Distance threshold for circle centers and maximum diameter constraint.

    Returns:
        list: A list of tuples (x, y, r) representing the circles.
    """
    circles = []
    max_attempts = 1000  # To avoid infinite loops if the constraints are too strict.

    while len(circles) < num_circles:
        for _ in range(max_attempts):
            r = np.random.uniform(min_radius, min(max_radius, threshold / 2))  # Radius limited by threshold
            x = np.random.uniform(r, 1 - r)  # x-coordinate ensuring the circle is within bounds
            y = np.random.uniform(r, 1 - r)  # y-coordinate ensuring the circle is within bounds

            candidate = (x, y, r)

            # Check constraints with existing circles
            if all(np.linalg.norm((x - cx, y - cy)) >= cr + r + threshold for cx, cy, cr in circles):
                circles.append(candidate)
                break
        else:
            raise ValueError("Cannot generate the required number of circles with the given constraints.")

    return circles


def generate_gaussians_with_constraints(num_gaussians, min_std, max_std, threshold):
    """
    Generate Gaussian distributions within the square [0, 1] x [0, 1] such that:
        - The squared distance between the centers of any two Gaussians plus the sum of their squared standard deviations
          is greater than the squared threshold.
        - Gaussians are well-separated, meaning they do not overlap within 1 standard deviation.

    Parameters:
        num_gaussians (int): Number of Gaussians to generate.
        min_std (float): Minimum standard deviation of the Gaussians.
        max_std (float): Maximum standard deviation of the Gaussians.
        threshold (float): Distance threshold for Gaussian centers and squared standard deviation constraint.

    Returns:
        list: A list of tuples (x, y, std) representing the Gaussians.
    """
    gaussians = []
    max_attempts = 1000  # To avoid infinite loops if the constraints are too strict.

    threshold_squared = threshold ** 2  # Precompute the squared threshold

    while len(gaussians) < num_gaussians:
        for _ in range(max_attempts):
            std = np.random.uniform(min_std, max_std)  # Standard deviation generation
            x = np.random.uniform(2 * std, 1 - 2 * std)  # x-coordinate ensuring the Gaussian is within bounds
            y = np.random.uniform(2 * std, 1 - 2 * std)  # y-coordinate ensuring the Gaussian is within bounds

            candidate = (x, y, std)

            # Check constraints with existing Gaussians
            # if all(
            #     # Ensure squared distance plus sum of squared stds is greater than threshold_squared
            #     (x - cx) ** 2 + (y - cy) ** 2 + 2 * (std ** 2 + cstd ** 2) > threshold_squared
            #     and
            #     # Ensure no overlap within 1 standard deviation
            #     # np.sqrt((x - cx) ** 2 + (y - cy) ** 2) > np.sqrt(2) *(std + cstd)
            #     np.abs(x - cx) > std + cstd
            #     and 
            #     np.abs(y - cy) > std + cstd
            #     for cx, cy, cstd in gaussians
            # ):
            if all(
                # Ensure Euclidean distance is greater than threshold + 2*std + 2*cstd
                np.sqrt((x - cx) ** 2 + (y - cy) ** 2) > threshold + 2 * std + 2 * cstd   # assumption: most of the points are within the 2 std-circle
                for cx, cy, cstd in gaussians
            ):
                gaussians.append(candidate)
                break
        else:
            raise ValueError("Cannot generate the required number of Gaussians with the given constraints.")

    return gaussians


def visualize_circles(circles, points, rand=''):
    """
    Visualize circles in the unit square [0, 1] x [0, 1].

    Parameters:
        circles (list): A list of tuples (x, y, r) representing the circles.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    
    # Draw the circles
    for i, (x, y, r) in enumerate(circles):
        circle = plt.Circle((x, y), r, fill=False, edgecolor='blue', linewidth=0.5, alpha=0.7)
        ax.scatter([point[0] for point in points[i]], [point[1] for point in points[i]], s=0.4)
        ax.add_patch(circle)
    
    # Draw the unit square boundary
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='black', linewidth=1.5)
    
    plt.title("Circles Visualization")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f'visualization_cluster{len(circles)}{rand}.png')


def sample_loc_from_truncated_gaussians(locations, variance=0.01, num_samples=1, lower_bound=0, upper_bound=1):
    """
    Sample points from multiple truncated Gaussian distributions centered around specified locations.

    Parameters:
        locations (list of lists): List of 2D points representing the centers of the Gaussians.
        variance (float): Shared variance of the Gaussian distributions.
        num_samples (int): Total number of samples to generate.
        lower_bound (float): Lower bound for truncation.
        upper_bound (float): Upper bound for truncation.

    Returns:
        numpy.ndarray: An array of shape (num_samples * len(locations), 2) containing the sampled points.
    """
    def truncnorm_sample(mean, variance, lower, upper, size):
        std_dev = np.sqrt(variance)
        a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
        return truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=size)

    samples = []

    for loc in locations:
        x_samples = truncnorm_sample(mean=loc[0], variance=variance, lower=lower_bound, upper=upper_bound, size=num_samples)
        y_samples = truncnorm_sample(mean=loc[1], variance=variance, lower=lower_bound, upper=upper_bound, size=num_samples)
        points = np.column_stack((x_samples, y_samples))
        samples.append(points.tolist())

    return np.array(samples)


def calculate_distance_matrix(circles):
    """
    Calculate the matrix of distances between the centers of circles minus the sum of their radii.

    Parameters:
        circles (list of tuples): A list of 3-tuples (x, y, r) representing the circles.

    Returns:
        numpy.ndarray: A matrix where the element (i, j) is the distance between the centers of circles i and j
                       minus the sum of their radii.
    """
    num_circles = len(circles)
    distance_matrix = np.zeros((num_circles, num_circles))
    
    for i in range(num_circles):
        for j in range(num_circles):
            if i != j:
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                center_distance = np.linalg.norm((x1 - x2, y1 - y2))
                distance_matrix[i, j] = center_distance - (r1 + r2)
    
    return distance_matrix


def get_args():
    parser = argparse.ArgumentParser(description='Generate synthetic data')
    parser.add_argument('--num-clusters-per-class', type=int, default=8, help='Number of clusters/gaussians to sample the intensities per class')
    parser.add_argument('--min-std', type=float, default=0.03, help='Min std for each gaussian')
    parser.add_argument('--max-std', type=float, default=0.06, help='Max std for each gaussian')
    parser.add_argument('--margin', type=float, default=0.005, help='Margin between the intra vs inter cluster distance')
    parser.add_argument('--max-intensity', type=float, default=100.0, help='Max intensity')
    parser.add_argument('--num-points-per-cluster', type=int, default=400)
    parser.add_argument('--randomize', default=True, action='store_true')
    parser.add_argument('--class-locs', type=str, default='[[0.1, 0.6], [0.4, 0.8]]')
    parser.add_argument('--spec-end', type=int, default=128)
    parser.add_argument('--spec-res', type=float, default=0.1)
    parser.add_argument('--peak-width', type=float, default=1.0)
    parser.add_argument('--noise-std', type=float, default=1.0)
    args = parser.parse_args()
    return args
# some other configs
# python create_synth_spec.py --num-clusters-per-class 4 --min-std 0.06 --max-std 0.08 --num-points-per-cluster 800
# python create_synth_spec.py --num-clusters-per-class 8 --min-std 0.03 --max-std 0.06 --num-points-per-cluster 400
# python create_synth_spec.py --num-clusters-per-class 16 --min-std 0.02 --max-std 0.04 --num-points-per-cluster 200
# python create_synth_spec.py --num-clusters-per-class 32 --min-std 0.01 --max-std 0.02 --num-points-per-cluster 100

if __name__ == '__main__':

    args = get_args()
    rand = '_rand' if args.randomize else ''
    args.class_locs = ast.literal_eval(args.class_locs)
    # args.num_points_per_cluster = int(args.num_points_per_cluster / 2)

    # 1. generate gaussian distributions as clusters of latents (i.e., peak intensities)
    gaussian_clusters = generate_gaussians_with_constraints(args.num_clusters_per_class, args.min_std, args.max_std, 2 * args.max_std + args.margin)
    dists = calculate_distance_matrix(gaussian_clusters)
    triu_dists = np.triu(dists, k=1)
    # print(np.min(triu_dists[triu_dists !=0]))
    
    # 2. sample the two peak intensities from the clusters
    sampled_intensities = {}
    sampled_points = []
    if args.randomize:
        total_points = args.num_points_per_cluster * args.num_clusters_per_class
        sampled_num_points = np.random.randint(10, total_points - 10, args.num_clusters_per_class - 1).tolist()
        sampled_num_points = [0] + sorted(sampled_num_points) + [total_points]
        sampled_num_points_per_cluster = [sampled_num_points[i+1] - sampled_num_points[i] for i in range(len(sampled_num_points) - 1)]
        print(sampled_num_points_per_cluster, sum(sampled_num_points_per_cluster))
    else:
        sampled_num_points_per_cluster = [args.num_points_per_cluster] * args.num_clusters_per_class
    for i in range(len(gaussian_clusters)):
        x, y, r = gaussian_clusters[i]
        sampled_points_in_circle = sample_points_from_gaussian(x, y, r, args.num_points_per_cluster) if not args.randomize else sample_points_from_gaussian(x, y, r, sampled_num_points_per_cluster[i])
        sampled_intensities[(x,y,r)] = [[args.max_intensity * x, args.max_intensity * y] for [x,y] in sampled_points_in_circle]
        sampled_points.append(sampled_points_in_circle)
    visualize_circles([(x,y,2*r) for (x,y,r) in gaussian_clusters], sampled_points, rand)

    # 3. sample the locations of the peaks for both classes
    sampled_locs = sample_loc_from_truncated_gaussians(locations=args.class_locs, num_samples=args.num_clusters_per_class * args.num_points_per_cluster)   # (num_classes, num_clusters * num_points, 2)
    sampled_locs = sampled_locs.reshape(sampled_locs.shape[0], args.num_clusters_per_class * args.num_points_per_cluster, sampled_locs.shape[-1]) # (num_classes, num_clusters, num_points, 2)
    
    # 4. generate mass spectra dict

    data_dict = {}
    for i in range(len(sampled_locs)):
        loc = tuple(args.class_locs[i])
        clus_specs = {}
        num_samples_so_far = 0
        for j in range(args.num_clusters_per_class):
            locs_this_cluster = sampled_locs[i, num_samples_so_far:num_samples_so_far + sampled_num_points_per_cluster[j], :]
            num_samples_so_far += sampled_num_points_per_cluster[j]
            print(locs_this_cluster.shape)
            circle_center = gaussian_clusters[j]
            intensities_this_cluster = sampled_intensities[circle_center]

            clus_specs[circle_center] = {'rel_intensities': [], 'spectra': []}
            clus_specs[circle_center]['rel_intensities'] = [[x[0] , x[1]] for x in intensities_this_cluster]
            peaks_list = [[[args.spec_end * locs_this_cluster[k][0], intensities_this_cluster[k][0], args.peak_width], [args.spec_end * locs_this_cluster[k][1], intensities_this_cluster[k][1], args.peak_width]] \
                for k in range(sampled_num_points_per_cluster[j])]
            for peaks in peaks_list:
                clus_specs[circle_center]['spectra'].append(generate_mass_spectrum(start=0, end=args.spec_end, resolution=args.spec_res, peaks=peaks, noise_std=args.noise_std)[-1])

        data_dict[loc] = clus_specs 

    with open(f'synth_specs_{args.num_points_per_cluster}{rand}.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
    




            



    