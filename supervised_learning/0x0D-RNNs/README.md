Architecture of a traditional RNN Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as inputs while having hidden states. They are typically as follows:

For each timestep ttt, the activation a<t>a^{< t >}a<t> and the output y<t>y^{< t >}y<t> are expressed as follows:
a<t>=g1(Waaa<t−1>+Waxx<t>+ba)andy<t>=g2(Wyaa<t>+by)\boxed{a^{< t >}=g_1(W_{aa}a^{< t-1 >}+W_{ax}x^{< t >}+b_a)}\quad\textrm{and}\quad\boxed{y^{< t >}=g_2(W_{ya}a^{< t >}+b_y)}a<t>=g1​(Waa​a<t−1>+Wax​x<t>+ba​)​andy<t>=g2​(Wya​a<t>+by​)​
where Wax,Waa,Wya,ba,byW_{ax}, W_{aa}, W_{ya}, b_a, b_yWax​,Waa​,Wya​,ba​,by​ are coefficients that are shared temporally and g1,g2g_1, g_2g1​,g2​ activation functions.

The pros and cons of a typical RNN architecture are summed up in the table below:

Advantages and 	Drawbacks
• Possibility of processing input of any length
• Model size not increasing with size of input
• Computation takes into account historical information
• Weights are shared across time 	• Computation being slow
• Difficulty of accessing information from a long time ago
• Cannot consider any future input for the current state
