import numpy as np
import torch.nn.functional as F

from dataloder import load_data , vec2eng , vec2de,add_start_token
from models import SimpleTransformer
import torch
import matplotlib.pyplot as plt


def main():
    train_df , test_df = load_data()
    contex_size = 300

    input_size = 300
    hidden_dim = 300
    batch_size = 1

    transformer = SimpleTransformer(input_size, contex_size, hidden_dim)
    optim = torch.optim.SGD(transformer.parameters(), lr=1e-2, momentum=0.9)
    losses = []
    counter = 0
    for idx, row in train_df.iterrows():
        de_sentence = row[0]  # German sentence vectors (list of vectors)
        eng_sentence = row[1]  # English sentence vectors (list of vectors)

        eng_sentence_with_start = add_start_token(eng_sentence)
        # Convert to tensors

        de_tensor = torch.tensor(de_sentence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        eng_tensor = torch.tensor(eng_sentence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        eng_tensor_with_start = torch.tensor(eng_sentence_with_start, dtype=torch.float32).unsqueeze(0)  # Add batch dimension


        # Forward pass
        pred = transformer.forward(de_tensor, eng_tensor_with_start)
        pred = pred.squeeze(0)  # Remove batch dimension
        #loss = torch.nn.MSELoss()
        target = eng_tensor
        #output = loss(pred, target)
        cos_sim = F.cosine_similarity(pred, target, dim=1)
        output = 1 - cos_sim.mean()
        print('output loss:', output.item())
        optim.zero_grad()
        output.backward()
        optim.step()
        losses.append(output.item())
        counter += 1
        if counter % 10 == 0:
            #print(f'Iteration {counter}, Loss: {output.item()}')
            #print('shape of pred:', np.shape(pred))
            for vec in pred:
                print(np.shape(vec))
                print(vec2eng(vec))

    # save the model
    torch.save(transformer.state_dict(), 'transformer_model.pth')
    # plot the losses
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations')
    plt.show()



if __name__ == '__main__':
    main()
