import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        weight = self.w
        return nn.DotProduct(weight, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        result = nn.as_scalar(self.run(x))
        if result >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            err = 0
            for x, y in dataset.iterate_once(1):
                pred = self.get_prediction(x)

                if nn.as_scalar(y) != pred:
                    self.w.update(x, -1*pred)
                    err = err + 1
            if err == 0:
                break




class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 10
        self.num_hidden_layers = 2
        self.hidden_layer_size = 15
        self.w = []
        self.b = []
        self.w.append(nn.Parameter(1, self.hidden_layer_size))
        self.w.append(nn.Parameter(self.hidden_layer_size, 10))
        self.w.append(nn.Parameter(10, 1))

        self.b.append(nn.Parameter(1, self.hidden_layer_size))
        self.b.append(nn.Parameter(1, 10))
        self.b.append(nn.Parameter(1, 1))


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        biased = None

        "*** YOUR CODE HERE ***"
        step = nn.Linear(x, self.w[0])
        biased = nn.AddBias(step, self.b[0])

        for i in range(self.num_hidden_layers):
            relu = nn.ReLU(biased)

            step = nn.Linear(relu, self.w[i + 1])

            biased = nn.AddBias(step, self.b[i + 1])
        return biased

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        cumulative_loss = 0
        cum_examples = 0
        for x, y in dataset.iterate_forever(self.batch_size):
            cum_examples = cum_examples + 1
            prediction = self.run(x)
            pred_loss = self.get_loss(x, y)
            cumulative_loss = cumulative_loss + nn.as_scalar(pred_loss)

            if cumulative_loss/cum_examples <= 0.02:
                print(cumulative_loss/cum_examples)
                break
            fullParams = []
            for w in self.w:
                fullParams.append(w)
            for b in self.b:
                fullParams.append(b)
            gradient = nn.gradients(pred_loss, fullParams)
            count = 0

            for w in self.w:
                w.update(gradient[count], -1*.25)
                count = count + 1
            for b in self.b:
                b.update(gradient[count], -1*.25)
                count = count + 1

        return prediction

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 10
        self.num_hidden_layers = 2
        self.hidden_layer_size = 200
        self.w = []
        self.b = []
        self.w.append(nn.Parameter(784, self.hidden_layer_size))
        self.w.append(nn.Parameter(self.hidden_layer_size, 80))
        self.w.append(nn.Parameter(80, 10))

        self.b.append(nn.Parameter(1, self.hidden_layer_size))
        self.b.append(nn.Parameter(1, 80))
        self.b.append(nn.Parameter(1, 10))


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        step = nn.Linear(x, self.w[0])
        biased = nn.AddBias(step, self.b[0])

        for i in range(self.num_hidden_layers):
            relu = nn.ReLU(biased)
            step = nn.Linear(relu, self.w[i + 1])
            biased = nn.AddBias(step, self.b[i + 1])

        return biased


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        acc = 0
        epoch = 0
        learning_rate = 0.35
        prediction = None
        while True:
            for x, y in dataset.iterate_once(self.batch_size):

                prediction = self.run(x)
                pred_loss = nn.SquareLoss(prediction, y)
                fullParams = self.w + self.b
                # for w in self.w:
                #     fullParams.append(w)
                # for b in self.b:
                #     fullParams.append(b)
                gradient = nn.gradients(pred_loss, fullParams)
                count = 0
                # if 10 < epoch < 20:
                #     learning_rate = 0.55
                # if 20 < epoch < 30:
                #     learning_rate = 0.3
                #
                # if epoch > 30:
                #     learning_rate = 0.01
                if epoch > 30:
                    learning_rate = 0.25
                for w in self.w:
                    w.update(gradient[count], -1 * learning_rate)
                    count = count + 1
                for b in self.b:
                    b.update(gradient[count], -1 * learning_rate)
                    count = count + 1
            epoch = epoch + 1
            acc = dataset.get_validation_accuracy()
            if acc > 0.92:
                learning_rate = 0.175
            if epoch > 20:
                learning_rate /= 10
            print(acc)
            print(epoch)
            if acc > 0.975:
                break
        return prediction


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 50
        self.hidden_layer_size = 25
        self.d = 150
        self.w_init = []
        self.w_init.append(nn.Parameter(self.num_chars, 200))
        self.w_init.append(nn.Parameter(200, self.d))
        self.w_hidden = nn.Parameter(self.d, self.d)
        # self.w_hidden = []
        # self.w_hidden.append(nn.Parameter(self.d, self.hidden_layer_size))
        # self.w_hidden.append(nn.Parameter(self.hidden_layer_size, self.d))
        self.b_init = []
        self.b_init.append(nn.Parameter(1, 200))
        self.b_init.append(nn.Parameter(1, self.d))
        self.w_final = nn.Parameter(self.d, 5)
        self.b = nn.Parameter(1, self.d)
        self.b_final = nn.Parameter(1, 5)



    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h = nn.Linear(xs[0], self.w_init[0])
        h = nn.AddBias(h, self.b_init[0])
        h = nn.ReLU(h)
        h = nn.Linear(h, self.w_init[1])
        h = nn.AddBias(h, self.b_init[1])
        h = nn.ReLU(h)
        # print('HELLO')

        for i in range(len(xs)):
            h = nn.ReLU(h)
            if i == 0:
                continue

            lin_hi = nn.Linear(h, self.w_hidden)
            lin_xi = nn.Linear(xs[i], self.w_init[0])
            lin_xi = nn.AddBias(lin_xi, self.b_init[0])
            lin_xi = nn.ReLU(lin_xi)
            lin_xi = nn.Linear(lin_xi, self.w_init[1])
            lin_xi = nn.AddBias(lin_xi, self.b_init[1])
            lin_xi = nn.ReLU(lin_xi)
            h = nn.AddBias(nn.Add(lin_hi, lin_xi), self.b)
        h_final = nn.Linear(h, self.w_final)
        h_final = nn.AddBias(h_final, self.b_final)
        return h_final

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learning_rate = 0.025
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                prediction = self.run(x)
                pred_loss = nn.SoftmaxLoss(prediction, y)
                fullParams = [self.w_init[0], self.w_init[1], self.w_hidden, self.w_final, self.b_init[0], self.b_init[1], self.b, self.b_final]
                gradient = nn.gradients(pred_loss, fullParams)
                self.w_init[0].update(gradient[0], -1 * learning_rate)
                self.w_init[1].update(gradient[1], -1 * learning_rate)
                self.w_hidden.update(gradient[2], -1 * learning_rate)
                self.w_final.update(gradient[3], -1 * learning_rate)
                self.b_init[0].update(gradient[4], -1 * learning_rate)
                self.b_init[1].update(gradient[5], -1 * learning_rate)
                self.b.update(gradient[6], -1 * learning_rate)
                self.b_final.update(gradient[7], -1 * learning_rate)
            accuracy = dataset.get_validation_accuracy()
            print(accuracy)
            if accuracy > 0.85:
                break
            if 0.75 < accuracy < 0.81:
                learning_rate = 0.01
            if accuracy >= 0.81:
                learning_rate = 0.005
            if accuracy > 0.83:

                learning_rate = 0.0007
        return prediction