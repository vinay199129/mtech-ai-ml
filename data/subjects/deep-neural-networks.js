/* Deep Neural Networks */
(window.COURSES = window.COURSES || []).push({
  slug: "deep-neural-networks",
  name: "Deep Neural Networks",
  semester: 1,
  desc: "From perceptrons to transformers: linear NNs, MLPs with backprop, CNNs (LeNet, AlexNet, VGG, Inception, ResNet) + transfer learning, RNN/LSTM/GRU/BiLSTM, attention, full Transformer (NLP & vision), modern optimization (SGD→Adam family), and regularization (dropout, batch/layer norm, weight decay).",
  textbooks: [
    "Zhang, Lipton, Li, Smola — Dive into Deep Learning (d2l.ai, FREE & interactive)",
    "Goodfellow, Bengio, Courville — Deep Learning (deeplearningbook.org, FREE)",
    "Chollet — Deep Learning with Python, 2e",
    "Jurafsky & Martin — Speech and Language Processing, 3e (FREE drafts)"
  ],
  modules: [
    {
      n: 1, title: "Introduction & Motivating Example",
      subtopics: ["What is deep learning, why now", "Wake-word detection example", "Key components: data, model, loss, optimizer", "Supervised learning examples"],
      ref: "T1 Ch. 1, R3 Ch. 1",
      notesFile: "01-intro.md",
      resources: [
        { type: "book", title: "d2l.ai — Chapter 1 (free, interactive)", url: "https://d2l.ai/chapter_introduction/index.html", desc: "Course textbook online — runs code in your browser." },
        { type: "video", title: "3Blue1Brown — But what is a neural network? (Ch. 1)", url: "https://www.youtube.com/watch?v=aircAruvnKk", desc: "Best visual intro to NNs ever made." },
        { type: "article", title: "Deep Learning Book — Ch. 1 (free)", url: "https://www.deeplearningbook.org/contents/intro.html", desc: "Goodfellow's framing of the field." }
      ]
    },
    {
      n: 2, title: "Artificial Neuron & Perceptron",
      subtopics: ["Biological vs artificial neuron", "Perceptron model", "Perceptron for AND/OR/NOT", "Perceptron learning algorithm", "XOR limitation (linearly separable only)"],
      ref: "Class notes",
      notesFile: "02-perceptron.md",
      resources: [
        { type: "video", title: "Perceptron explained (Welch Labs)", url: "https://www.youtube.com/watch?v=4Gac5I64LM4", desc: "Geometric picture of perceptron decision boundary." },
        { type: "article", title: "Rosenblatt's perceptron — Andrew Ng CS229 notes", url: "https://cs229.stanford.edu/notes2022fall/main_notes.pdf", desc: "Quick formal treatment + convergence theorem." }
      ]
    },
    {
      n: 3, title: "Linear Neural Network for Regression",
      subtopics: ["Single neuron, no hidden layers", "Squared loss", "Batch gradient descent training", "Inference"],
      ref: "T1 Ch. 3 & 12",
      notesFile: "03-linear-nn-regression.md",
      resources: [
        { type: "book", title: "d2l.ai — Linear Regression (Ch. 3)", url: "https://d2l.ai/chapter_linear-regression/index.html", desc: "From-scratch + framework implementation." },
        { type: "video", title: "Andrew Ng — Linear regression as a single neuron", url: "https://www.youtube.com/watch?v=kHwlB_j7Hkc", desc: "Frame regression as the simplest NN." }
      ]
    },
    {
      n: 4, title: "Linear Neural Network for Classification",
      subtopics: ["Sigmoid + binary cross-entropy", "SGD training", "Multi-class: softmax + cross-entropy", "Mini-batch SGD"],
      ref: "T1 Ch. 4 & 12",
      notesFile: "04-linear-nn-classification.md",
      resources: [
        { type: "book", title: "d2l.ai — Linear Classification (Ch. 4)", url: "https://d2l.ai/chapter_linear-classification/index.html", desc: "Aligned with handout's softmax + cross-entropy lab." },
        { type: "video", title: "Softmax + cross-entropy explained", url: "https://www.youtube.com/watch?v=ErfnhcEV1O8", desc: "Why softmax pairs with cross-entropy." }
      ]
    },
    {
      n: 5, title: "Deep Feedforward Networks (MLPs) & Backprop",
      subtopics: ["XOR via hidden layers", "Activation functions: sigmoid, tanh, ReLU, GELU", "Forward pass (vectorized)", "Backpropagation (vectorized)", "Width vs depth"],
      ref: "T1 Ch. 5",
      notesFile: "05-mlp-backprop.md",
      resources: [
        { type: "video", title: "3Blue1Brown — NN series (Ch. 1–4)", url: "https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi", desc: "Includes the famous backprop visualization." },
        { type: "book", title: "d2l.ai — MLPs (Ch. 5)", url: "https://d2l.ai/chapter_multilayer-perceptrons/index.html", desc: "Forward + backward in code." },
        { type: "article", title: "Karpathy — A recipe for training neural networks", url: "http://karpathy.github.io/2019/04/25/recipe/", desc: "Practical playbook to actually make MLPs work." }
      ]
    },
    {
      n: 6, title: "Convolutional Neural Networks",
      subtopics: ["Image data, locality, translation invariance", "Convolution operation, kernel learning", "Padding, stride, channels", "1×1 convolution", "Pooling", "LeNet"],
      ref: "T1 Ch. 7, R3 §2.2.11",
      notesFile: "06-cnn.md",
      resources: [
        { type: "course", title: "Stanford CS231n — Convolutional Neural Networks", url: "http://cs231n.stanford.edu/", desc: "Best CNN course on the internet — notes are open." },
        { type: "book", title: "d2l.ai — CNNs (Ch. 7)", url: "https://d2l.ai/chapter_convolutional-neural-networks/index.html", desc: "Builds LeNet from scratch." },
        { type: "article", title: "CNN Explainer (interactive)", url: "https://poloclub.github.io/cnn-explainer/", desc: "Click-through visualization of what each layer does." }
      ]
    },
    {
      n: 7, title: "Deep CNN Architectures (AlexNet, VGG, NiN, Inception, ResNet)",
      subtopics: ["AlexNet & representation learning", "VGG blocks", "NiN blocks", "Inception (GoogLeNet)", "Residual learning (ResNet)"],
      ref: "T1 Ch. 8",
      notesFile: "07-cnn-architectures.md",
      resources: [
        { type: "book", title: "d2l.ai — Modern CNNs (Ch. 8)", url: "https://d2l.ai/chapter_convolutional-modern/index.html", desc: "All five architectures implemented and compared." },
        { type: "article", title: "An Overview of ResNet and its Variants (TDS)", url: "https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035", desc: "Why residuals work + variants." },
        { type: "video", title: "CS231n — CNN Architectures lecture", url: "https://www.youtube.com/watch?v=DAOcjicFr1Y", desc: "AlexNet → ResNet historical walkthrough." }
      ]
    },
    {
      n: 8, title: "Transfer Learning & Fine-Tuning",
      subtopics: ["Pretrained backbones", "Feature extraction vs fine-tuning", "When/which layers to freeze", "Domain adaptation basics"],
      ref: "Class notes",
      notesFile: "08-transfer-learning.md",
      resources: [
        { type: "docs", title: "PyTorch — Transfer learning tutorial", url: "https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html", desc: "End-to-end ResNet18 fine-tuning example." },
        { type: "docs", title: "TensorFlow/Keras — Transfer learning & fine-tuning", url: "https://www.tensorflow.org/tutorials/images/transfer_learning", desc: "Same pattern in TF for the lab." }
      ]
    },
    {
      n: 9, title: "Recurrent Neural Networks",
      subtopics: ["Sequence data, raw text → tokens", "RNN with hidden state", "Backprop through time (BPTT)", "Encoder–decoder", "Teacher forcing", "Loss masking"],
      ref: "T1 Ch. 9, R4 Ch. 9",
      notesFile: "09-rnn.md",
      resources: [
        { type: "article", title: "Karpathy — The Unreasonable Effectiveness of RNNs", url: "http://karpathy.github.io/2015/05/21/rnn-effectiveness/", desc: "The article that hooked a generation on RNNs." },
        { type: "book", title: "d2l.ai — RNNs (Ch. 9)", url: "https://d2l.ai/chapter_recurrent-neural-networks/index.html", desc: "Implements RNNs from scratch." },
        { type: "video", title: "StatQuest — RNN clearly explained", url: "https://www.youtube.com/watch?v=AsNTP8Kwu80", desc: "Visual walkthrough of unrolling and BPTT." }
      ]
    },
    {
      n: 10, title: "LSTM, GRU, BiLSTM, Stacked RNNs",
      subtopics: ["Vanishing/exploding gradients in RNNs", "LSTM gates (input, forget, output)", "GRU (reset, update)", "Bidirectional RNNs", "Deep / stacked RNNs"],
      ref: "T1 Ch. 10, R4 Ch. 9",
      notesFile: "10-lstm-gru.md",
      resources: [
        { type: "article", title: "Christopher Olah — Understanding LSTM Networks", url: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/", desc: "Iconic blog post — read this before anything else." },
        { type: "video", title: "StatQuest — LSTM clearly explained", url: "https://www.youtube.com/watch?v=YCzL96nL7j0", desc: "Same gates, animated." },
        { type: "book", title: "d2l.ai — Modern RNNs (Ch. 10)", url: "https://d2l.ai/chapter_recurrent-modern/index.html", desc: "LSTM/GRU/BiRNN with code." }
      ]
    },
    {
      n: 11, title: "Attention Mechanism",
      subtopics: ["Queries, keys, values", "Attention pooling (Nadaraya–Watson)", "Dot-product & additive attention", "Scaled dot-product attention", "Bahdanau attention", "Multi-head attention", "Self-attention", "Positional encoding"],
      ref: "T1 Ch. 11, R4 Ch. 9",
      notesFile: "11-attention.md",
      resources: [
        { type: "article", title: "Jay Alammar — The Illustrated Transformer (covers attention)", url: "https://jalammar.github.io/illustrated-transformer/", desc: "Single best illustrated explanation of QKV attention." },
        { type: "video", title: "Yannic Kilcher — Attention Is All You Need (paper read)", url: "https://www.youtube.com/watch?v=iDulhoQ2pro", desc: "Walks through the paper section by section." },
        { type: "book", title: "d2l.ai — Attention Mechanisms (Ch. 11)", url: "https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html", desc: "From-scratch attention implementations." }
      ]
    },
    {
      n: 12, title: "Transformer Architecture",
      subtopics: ["Encoder block, decoder block", "Position-wise FFN", "Residual connections + LayerNorm", "Masked self-attention", "Putting it together"],
      ref: "T1 Ch. 11, R4 Ch. 10",
      notesFile: "12-transformer.md",
      resources: [
        { type: "article", title: "The Annotated Transformer (Harvard NLP)", url: "http://nlp.seas.harvard.edu/2018/04/03/attention.html", desc: "Paper + working PyTorch code line-by-line." },
        { type: "article", title: "Jay Alammar — Illustrated Transformer", url: "https://jalammar.github.io/illustrated-transformer/", desc: "Companion to the paper read above." },
        { type: "video", title: "Andrej Karpathy — Let's build GPT from scratch", url: "https://www.youtube.com/watch?v=kCc8FmEb1nY", desc: "2-hour live coding of a full transformer. Worth every minute." }
      ]
    },
    {
      n: 13, title: "Vision Transformers & Pretraining at Scale",
      subtopics: ["Patch embeddings", "ViT encoder", "Encoder-only / encoder-decoder / decoder-only families", "Scalability laws", "Pretraining → fine-tuning"],
      ref: "T1 Ch. 11, R4 §10.7",
      notesFile: "13-vit-pretraining.md",
      resources: [
        { type: "article", title: "ViT — An Image is Worth 16x16 Words (paper)", url: "https://arxiv.org/abs/2010.11929", desc: "Original ViT paper, very readable." },
        { type: "article", title: "Hugging Face — Transformer family overview", url: "https://huggingface.co/docs/transformers/model_summary", desc: "Concise map of BERT/GPT/T5/ViT and friends." },
        { type: "video", title: "Yannic Kilcher — ViT explained", url: "https://www.youtube.com/watch?v=TrdevFK_am4", desc: "Paper read + critique." }
      ]
    },
    {
      n: 14, title: "Optimization for Deep Models",
      subtopics: ["GD, SGD, mini-batch SGD", "Momentum, Nesterov", "AdaGrad, RMSProp, AdaDelta", "Adam", "Learning rate schedules"],
      ref: "T1 Ch. 12",
      notesFile: "14-deep-optimization.md",
      resources: [
        { type: "article", title: "Distill — Why Momentum Really Works", url: "https://distill.pub/2017/momentum/", desc: "Best interactive treatment of optimizer dynamics." },
        { type: "article", title: "Sebastian Ruder — An overview of gradient descent", url: "https://www.ruder.io/optimizing-gradient-descent/", desc: "Adam, RMSProp, AdaGrad with formulas." },
        { type: "book", title: "d2l.ai — Optimization (Ch. 12)", url: "https://d2l.ai/chapter_optimization/index.html", desc: "Side-by-side code for every optimizer." }
      ]
    },
    {
      n: 15, title: "Regularization for Deep Models",
      subtopics: ["Underfitting vs overfitting", "Train/test error & generalization gap", "Weight decay (L2)", "Dropout", "Batch normalization", "Layer normalization", "Distribution shift"],
      ref: "T1 §3.6, 3.7, 4.6, 4.7, 5.5, 5.6, 8.5, 11.7",
      notesFile: "15-deep-regularization.md",
      resources: [
        { type: "article", title: "Dropout: A Simple Way to Prevent Overfitting (Srivastava et al.)", url: "https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf", desc: "Original dropout paper — short & exam-ready." },
        { type: "article", title: "Batch Normalization paper (Ioffe & Szegedy)", url: "https://arxiv.org/abs/1502.03167", desc: "Original BN paper." },
        { type: "video", title: "Andrew Ng — Regularization for deep learning (full mini-course)", url: "https://www.youtube.com/watch?v=NyG-7nRpsW8", desc: "Covers dropout, BN, weight decay, early stopping." }
      ]
    }
  ]
});
