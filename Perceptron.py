from manim import *


config.pixel_height = 1080
config.pixel_width = 1920
config.frame_height = 8.0
config.frame_width = 14.0
config.frame_rate=60


class HeavisideFunction(Scene):
    def construct(self):
        # Define the Heaviside function as a MathTex object
        heaviside_tex = MathTex(r"\text{Heaviside}(x) = \begin{cases} 0 & \text{if } x < 0 \\ 1 & \text{if } x \geq 0 \end{cases}").scale(0.7)
        heaviside_tex.to_edge(UP)
        heaviside_tex.set_color_by_gradient(RED, BLUE)

        # Show the Heaviside function text
        self.play(Write(heaviside_tex))

        # Create X and Y axes
        axes = Axes(
            x_range=[-4, 4, 1],  # x-axis from -4 to 4 with step of 1
            y_range=[-0.5, 1.5, 0.5],  # y-axis from -0.5 to 1.5 with step of 0.5
            tips=True,
            axis_config={"include_numbers": True},
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="Heaviside(x)")

        # Show axes
        self.play(Create(axes), Write(axes_labels))

        # Define the Heaviside function points
        points = [
            (-3, 0),
            (-2, 0),
            (-1, 0),
            (0, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
        ]

        # Create dots at the function points
        dots = VGroup(*[Dot(color=RED).move_to(axes.c2p(x, y)) for x, y in points])

        # Create lines connecting the dots
        lines = VGroup()
        for i in range(len(points) - 1):
            if points[i][0] < 0 and points[i + 1][0] == 0:  # Skip jump at x = 0
                continue
            lines.add(Line(axes.c2p(*points[i]), axes.c2p(*points[i + 1]), color=BLUE))

        # Show the dots and lines
        self.play(Create(dots), Create(lines))
        axis = VGroup(axes,lines,dots,axes_labels)
        self.play(axis.animate.scale(0.6).move_to(ORIGIN))

        # Add explanation text and adjust for overlap
        explanation_text = Text(
            "The Heaviside function is 0 for x < 0 and 1 for x ≥ 0",
            font_size=24
        )
        explanation_text.next_to(axes, DOWN, buff=0.5)

        self.play(Write(explanation_text))

        # Highlight the jump at x = 0
        jump_text = Text("Jump at x = 0", font_size=20, color=YELLOW)
        jump_text.next_to(axes.c2p(0, 1), RIGHT+UP, buff=0.5)

        self.play(Write(jump_text))

        self.wait(2)

class NeuralNetwork(Scene):
    def draw_line(self, start, end, extra, label=None, label_buff=0.2, return_line=False):
        vec = end - start
        norm_vec = np.linalg.norm(vec)
        unit_vec = vec / norm_vec
        end = start + unit_vec * (norm_vec - extra)

        line = Line(start, end)
        self.play(Create(line))
        if label:
            label.next_to(line, UP, buff=label_buff)
            self.play(Write(label))
        return line if return_line else None

    def create_matrix(self, matrix_data, row_labels, col_labels, scale=0.5):
        table = Table(
            matrix_data,
            include_outer_lines=True,
            row_labels=[Text(str(label)).scale(0.8) for label in row_labels],
            col_labels=[Text(str(label)).scale(0.8) for label in col_labels],
            line_config={"color": BLUE}
        ).scale(scale)
        return table

    def construct(self):
        # Shift everything left to make room for matrix
        shift_left = LEFT * 3

        # Neural Network Part
        dot_text = Text("X1,X2,X3").scale(0.5)
        neuron_text = Text("Neuron").scale(0.5)

        circle = Circle(radius=0.8).set_color(BLUE)
        dots = VGroup(*[Dot() for _ in range(3)]).arrange(RIGHT, buff=1.8)
        circle.shift(DOWN * 0.8 + shift_left)
        dots.shift(DOWN * 2.5 + shift_left)
        weight_labels = VGroup(*[MathTex(f"W_{i+1}").scale(0.6) for i in range(3)])

        self.play(ReplacementTransform(dot_text, dots))
        neuron_text.next_to(circle, UP)
        self.play(Write(neuron_text))
        self.play(ReplacementTransform(neuron_text, circle))

        for dot, weight_label in zip(dots, weight_labels):
            self.draw_line(dot.get_center(), circle.get_center(), circle.radius, weight_label)

        output_dot = Dot().scale(0.5)
        output_dot.next_to(circle, UP, buff=1)
        output_text = Text("Output").scale(0.7).next_to(output_dot, RIGHT)
        self.play(Create(output_text))
        self.play(ReplacementTransform(output_text, output_dot))

        # Output connection
        output_weight = Text('w (final layer)').scale(0.5)
        output_line = self.draw_line(output_dot.get_center(), circle.get_center(), circle.radius, return_line=True)
        output_weight.next_to(output_line, RIGHT, buff=0.5)
        self.play(Create(output_weight))

        # Labels
        y_hat = MathTex(r"\hat{y}").scale(0.8)
        y_hat.next_to(output_dot, UP, buff=0.8)
        self.play(Write(y_hat))

        output_eq = MathTex(r"Output = Heaviside(\hat{y})").scale(0.8)
        output_eq.next_to(y_hat, RIGHT, buff=0.8)
        self.play(Create(output_eq))

        # Matrix Part
        input_matrix = [
            ['120', '3.8', '95'],
            ['115', '3.6', '88'],
            ['125', '3.9', '92'],
            ['118', '3.7', '90'],
            ['122', '3.8', '93']
        ]
        
        weight_matrix = [
            ['0.5'],
            ['0.3'],
            ['0.2']
        ]

        # Create input matrix
        input_table = self.create_matrix(
            input_matrix,
            row_labels=['1', '2', '3', '4', '5'],
            col_labels=['IQ', 'CGPA', 'Score'],
            scale=0.4
        )
        input_table.to_edge(RIGHT + UP)

        # Create weight matrix
        weight_table = self.create_matrix(
            weight_matrix,
            row_labels=['w1', 'w2', 'w3'],
            col_labels=['Weight'],
            scale=0.4
        )
        weight_table.next_to(input_table, DOWN, buff=0.5)

        # Matrix multiplication symbol
        mult_symbol = MathTex(r"\times").scale(0.8)
        mult_symbol.next_to(input_table, RIGHT, buff=0.3)

        # Show matrices and multiplication
        self.play(Create(input_table))
        self.play(Create(weight_table))
        self.play(Write(mult_symbol))

        # Highlight the first row
        first_row_highlight = SurroundingRectangle(input_table.get_rows()[1], color=YELLOW, buff=0.05)
        self.play(Create(first_row_highlight))

        # Show matrix multiplication explanation
        multiplication = MathTex(
            r"\hat{y} = w_1 \cdot {120} + w_2 \cdot 3.8 + w_3 \cdot 95"
        ).scale(0.7)
        multiplication.move_to(ORIGIN+RIGHT*0.3).scale(1)
        self.play(Write(multiplication))

        # Transform to y_hat
        y_hat_equation = MathTex(r"\hat{y} = 60 + 1.14 + 19 = 80.14").scale(0.8)
        y_hat_equation.next_to(multiplication, DOWN, buff=0.2)
        self.play(ReplacementTransform(multiplication, y_hat_equation))

        # Transform to Heaviside output
        heaviside_equation = MathTex(r"Output = Heaviside(80.14) = 1").scale(0.7)
        heaviside_equation.next_to(y_hat_equation, DOWN, buff=0.5)
        self.play(ReplacementTransform(y_hat_equation,heaviside_equation))

        # Explanation text
        explanation = MathTex(r"\text{Inputs} \times \text{Weights} \to \hat{y} \to \text{Output}").scale(0.8)

        explanation.to_edge(DOWN)
        self.play(Write(explanation))

        self.wait(2)


class BoxAnimation(Scene):
    
    def create_matrix(self, matrix_data, row_labels, col_labels, scale=0.5):
        table = Table(
            matrix_data,
            include_outer_lines=True,
            row_labels=[Text(str(label)).scale(0.8) for label in row_labels],
            col_labels=[Text(str(label)).scale(0.8) for label in col_labels],
            line_config={"color": BLUE}
        ).scale(scale)
        return table

    def draw_lines(self, to, dots, extra=0, **kwargs):
        # Calculate line start and end points
        start = dots.get_center()
        end = to.get_center()
        vec = end - start
        vec_norm = np.linalg.norm(vec)
        unit_vec = vec / vec_norm
        end = start + unit_vec * (vec_norm - extra)

        # Create and animate the line
        line = Line(start, end)
        self.play(Create(line))

        # Handle optional label
        label = kwargs.get('label', None)
        if label:
            label.next_to(line, UP, buff=kwargs.get('label_buff', 0.2))
            self.play(Write(label))

        return line if kwargs.get('return_line', False) else None

    def heaviside(self,x,thres=0.5):
        return 1 if x>thres else 0

    def construct(self):
        width, height = 5, 3
        perceptron_text = Text("Perceptron").scale(0.5)
        outer = Rectangle(width=width, height=height).set_color(BLUE).set_fill(BLUE, opacity=0.5)
        inner = Rectangle(width=width / 2, height=height / 2).set_color(YELLOW).set_fill(BLACK, opacity=1)

        # Position the inner rectangle and text within the outer rectangle
        inner.move_to(outer.get_center())
        perceptron_text.move_to(inner.get_center())
        
        perceptron_rec = VGroup(outer,inner,perceptron_text)

        # Group the outer and inner rectangles and text as a single perceptron object
        self.add(perceptron_rec)
        self.play(Create(perceptron_rec))
        
        # Create inputs and arrange them vertically with spacing
        inputs = VGroup(*[Dot() for _ in range(3)]).arrange(UP, buff=1)
        
        # Create y_hat vector
        y_hat = MathTex(r"\hat{y}").scale(1.5).next_to(perceptron_rec, RIGHT, buff=1.4)
        
        # Random output weights
        output_weights = [0.5,0.3,0.2]
        
        # Add inputs and animate them to the left
        self.add(inputs.scale(1.2))
        self.play(inputs.animate.move_to(LEFT * width))

        # Draw lines from each input dot to the outer rectangle
        input_lines = VGroup()
        for input_dot in inputs:
             input_lines.add(self.draw_lines(outer, input_dot, extra=width / 2, label=False, return_line=True))

        # Add y_hat and animate
        self.add(y_hat)

        perceptron = Group(perceptron_rec, input_lines, inputs, y_hat)
        self.add(perceptron)
        
        self.play(perceptron.animate.move_to(LEFT+UP).scale(0.6))

        table_data = [
            ['120', '3.8', '95','1'],
            ['115', '3.6', '88','0'],
            ['125', '3.9', '92','1'],
            ['118', '3.7', '90','1'],
            ['122', '3.8', '93','0']
        ]
        
        # Create the table
        table = self.create_matrix(
            table_data,
            row_labels=['1', '2', '3', '4', '5'],
            col_labels=['IQ', 'CGPA', 'Score','Placement'],
            scale=0.3
        )
        
        # Position the table
        table.to_corner(RIGHT*0.3)
        
        # Add the table to the scene
        self.play(Create(table))
        self.wait(1)
        
        # Create initial matrix
        matrix = MathTex(r"\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}")
        matrix.scale(1).move_to(LEFT+DOWN)
        self.play(Write(matrix))
        self.wait(1)
        
        arrow_to_yhat = Arrow(
            start=perceptron_rec.get_right(),
            end=y_hat.get_center()-RIGHT*0.4,
            buff=0.1,
            color=WHITE
        ).scale(1.1)
        self.play(Create(arrow_to_yhat))

        input_label = Text("inputs=").scale(0.7).next_to(matrix, LEFT, buff=0.3)
        self.play(Write(input_label))

        perceptron_formula = Text(r"\hat{y} = Perceptron(inputs)").scale(0.4).move_to(DOWN*2.5)
        self.play(Write(perceptron_formula))
        # Highlight all rows except the header
        rows_to_highlight = table.get_rows()[1:]
        
        for row_index, row in enumerate(rows_to_highlight):
            

            highlighted_row = row.copy().set_color(YELLOW).scale(1.2)
            
            current_matrix = MathTex(
        rf"\begin{{bmatrix}} {table_data[row_index][0]} \\ {table_data[row_index][1]} \\ {table_data[row_index][2]} \end{{bmatrix}}"
            ).scale(1).move_to(input_label.get_center() + RIGHT * 2)

    # Animate the row highlight and matrix transformation
            self.play(
                ReplacementTransform(row, highlighted_row),  # Highlight current row
                ReplacementTransform(matrix, current_matrix)  # Transform matrix
            )

            # Update matrix reference
            matrix = current_matrix

            # Calculate weighted sum with random weights
            weighted_sum = sum(
                float(table_data[row_index][j]) * output_weights[j] for j in range(3)
            )
            weighted_sum = self.heaviside(weighted_sum)
            new_perceptron_formula = MathTex(
                rf"\hat{{y}} = \text{{Perceptron}}\left({{{table_data[row_index][0]}, {table_data[row_index][1]}, {table_data[row_index][2]}}}\right)"
            ).scale(0.7).move_to(perceptron_formula.get_center())

            self.play(ReplacementTransform(perceptron_formula, new_perceptron_formula))
            perceptron_formula = new_perceptron_formula
            # Update y_hat with weighted sum
            
            y_hat_value = MathTex(rf"{weighted_sum:.2f}").scale(0.5).move_to(y_hat.get_center())


            self.play(ReplacementTransform(y_hat, y_hat_value))

            y_hat = y_hat_value

            self.wait(1)

        # Optional: Reset table to original state
        self.play(
            *[row.animate.set_color(WHITE) for row in table.get_rows()[1:]]
        )
        self.wait(1)

class Accuracy(Scene):
    def construct(self):
        # Titles
        pred_title = Text("Prediction:").scale(0.7).move_to(UP * 2.5 + LEFT * 3)
        act_title = Text("Actual:").scale(0.7).move_to(UP * 2.5 + RIGHT * 3)
        
        # Data
        predictions = [1, 1, 1, 1, 1]
        actuals = [1, 0, 1, 1, 0]
        
        # Convert data to MathTex rows
        pred_tex = MathTex(
            f"{predictions}"
        ).next_to(pred_title, DOWN, buff=0.5).scale(0.7)
        act_tex = MathTex(
            f"{actuals}"
        ).next_to(act_title, DOWN, buff=0.5).scale(0.7)
        
        # Accuracy formula
        accuracy_formula = MathTex(r"\text{Accuracy} = \frac{\text{Correct}}{\text{Total}} = 0").scale(0.8).to_edge(DOWN)
        
        # Draw titles and initial lists
        self.play(Write(pred_title), Write(act_title))
        self.play(Write(pred_tex), Write(act_tex))
        self.play(Write(accuracy_formula))
        
        # Create a rectangle to hover over elements
        rect_pred = SurroundingRectangle(pred_tex[0][1], color=YELLOW, buff=0.15)
        rect_act = SurroundingRectangle(act_tex[0][1], color=YELLOW, buff=0.15)
        self.play(Create(rect_pred), Create(rect_act))
        
        # Count correct predictions
        correct = 0
        total = len(predictions)
        
        # Loop through predictions and actuals
        for i in range(total):
            # Highlight current elements
            self.play(
                rect_pred.animate.move_to(pred_tex[0][i * 2 + 1]),
                rect_act.animate.move_to(act_tex[0][i * 2 + 1]),
                run_time=0.5,
            )
            
            # Check correctness and update accuracy
            if predictions[i] == actuals[i]:
                correct += 1
            
            # Update the accuracy formula
            accuracy_formula_new = MathTex(
                rf"\text{{Accuracy}} = \frac{{\text{{Correct}}}}{{\text{{Total}}}} = \frac{{{correct}}}{{{total}}} = {correct / total:.2f}"
            ).scale(0.8).to_edge(DOWN)
            self.play(Transform(accuracy_formula, accuracy_formula_new))
        
        self.wait(2)


class ScientistsAnimation(Scene):
    def construct(self):
        # Warren McCulloch details
        warren_name = Text("Warren McCulloch", font_size=36).to_edge(LEFT+UP, buff=1)
        warren_passion = Text("American Neuropsychologist", font_size=24).next_to(warren_name, DOWN, aligned_edge=LEFT+UP)
        warren_image = ImageMobject(r"C:\Users\Heavenly\Downloads\warren mcculloch.jpg").scale(2.5).next_to(warren_passion, DOWN, aligned_edge=LEFT+UP)
        warren_dates = Text("[Nov 16 1898 - Sep 24 1969]", font_size=24).next_to(warren_image, DOWN, aligned_edge=LEFT)

        warren_group = Group(warren_name, warren_passion, warren_image, warren_dates)

        # Walter Pitts details
        walter_name = Text("Walter Pitts", font_size=36).to_edge(RIGHT+UP, buff=1)
        walter_passion = Text("Mathematician & Logician", font_size=24).next_to(walter_name, DOWN, aligned_edge=RIGHT+UP)
        walter_image = ImageMobject(r"C:\Users\Heavenly\Downloads\walter pitts.jpg").scale(0.5).next_to(walter_passion, DOWN, aligned_edge=RIGHT+UP)
        walter_dates = Text("[Apr 23 1923 - May 14 1969]", font_size=24).next_to(walter_image, DOWN, aligned_edge=RIGHT)

        walter_group = Group(walter_name, walter_passion, walter_image, walter_dates)

        # Animation for Warren
        self.play(FadeIn(warren_group))
        self.wait(2)
        self.play(FadeIn(walter_group))
        self.wait(2)
        self.wait(2)

class SingleScientist(Scene):
    def __init__(self,**kwargs):
        super(SingleScientist,self).__init__(**kwargs)
        self.name = 'Frank Rosenblatt'
        self.path = r"C:\Users\Heavenly\Downloads\frank rosenbalt.jpg"

    def construct(self):
        scientist = Text(self.name, font_size=36).to_edge(ORIGIN+UP, buff=1)
        scientist_passion = Text("American Neuropsychologist", font_size=24).next_to(scientist, DOWN, aligned_edge=LEFT+UP)
        scientist_image = ImageMobject(self.path).scale(0.8).next_to(scientist_passion, DOWN, aligned_edge=LEFT+UP)
        scientist_dates = Text("[Nov 16 1898 - Sep 24 1969]", font_size=24).next_to(scientist_image, DOWN, aligned_edge=LEFT)

        scientist_group = Group(scientist, scientist_passion, scientist_image, scientist_dates)

        name = Text("Cornell University", font_size=36).to_edge(ORIGIN+UP, buff=1)
        lab_image = ImageMobject(r"C:\Users\Heavenly\Downloads\cornell.jpg").scale(0.7).next_to(name, DOWN, aligned_edge=RIGHT+UP)
        dates = Text("1957", font_size=24).next_to(lab_image, DOWN, aligned_edge=ORIGIN)

        group = Group(name, lab_image, dates)



        self.play(FadeIn(scientist_group))
        self.play(scientist_group.animate.to_edge(LEFT))
        self.wait(2)
        self.play(FadeIn(group))
        self.play(group.animate.to_edge(RIGHT))
        self.wait(2)
        
class ImageScene(Scene):
    def construct(self):
        name = Text("IBM 704 Computer", font_size=36).to_edge(ORIGIN+UP, buff=1)
        lab_image = ImageMobject(r"C:\Users\Heavenly\Downloads\IBM 704.jpg").scale(1).next_to(name, DOWN, aligned_edge=RIGHT+UP)
        dates = Text("1957", font_size=24).next_to(lab_image, DOWN, aligned_edge=ORIGIN)

        group = Group(name, lab_image, dates)



        self.play(FadeIn(group),subcaption_offset=1)
        self.wait(2)

        