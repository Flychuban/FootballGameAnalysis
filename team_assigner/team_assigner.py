from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
    
    def get_clustering_model(self):
        # Reshape image to 2D array
        image = image.reshape(-1, 3)
        
        # Perform KMeans with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=0).fit(image)
        
        return kmeans
    
    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2),:]
        
        # Get clustering model
        kmeans = self.get_clustering_model()
        
        # Get cluster labels for each pixel
        labels = kmeans.labels_
        
        # Reshape labels to image shape
        labels = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        
        # Get the palyer cluster
        corner_clusters = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 if non_player_cluster == 0 else 0
        
        # Get the player color
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color
        
    
    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detections in player_detections.items():
            bbox = player_detections['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(player_colors)
        
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]