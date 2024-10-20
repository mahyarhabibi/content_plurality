import numpy as np
import os

def load_embeddings(emb_dir, ids_name='ids', vecs_name='embeddings', max_n=1_000_000, keep_ids=None):
    """
    Load and return stored embeddings in emb_dir
    max_n: maximum number of unique embeddings to load
    keep_ids: list of tweet_ids to keep (if None, keep all)
    """
    files = [os.path.join(emb_dir, path) for path in os.listdir(emb_dir) 
             if path.endswith('.npz')]

    # Use a dictionary to store unique IDs and their corresponding embeddings
    id_vec_dict = {}

    try:
        # Iterate through each file, loading the arrays and updating the dictionary
        for path in files:
            array = np.load(path)
            ids = array[ids_name]
            vecs = array[vecs_name]

            if keep_ids is not None:
                # Create a boolean mask for ids that are in keep_ids
                mask = np.isin(ids, keep_ids)
                ids = ids[mask]
                vecs = vecs[mask]

            # Update the dictionary with new unique IDs and their embeddings
            for id, vec in zip(ids, vecs):
                if id not in id_vec_dict and len(id_vec_dict) < max_n:
                    id_vec_dict[id] = vec
                
            # Break if we've reached max_n unique IDs
            if len(id_vec_dict) >= max_n:
                break

    except Exception as e:
        print(f"Error occurred while processing {path}: {e}")

    # Convert the dictionary to numpy arrays
    arr_id = np.array(list(id_vec_dict.keys()))
    tweet_embd = np.array(list(id_vec_dict.values()))

    return arr_id, tweet_embd