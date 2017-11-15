

class Forest:
    def __init__(self, row, col, init_pos):

        self.row_num = row
        self.col_num = col
        self.cell_num = row*col
        self.cells = [{} for i in range(row * col)]
        for i in range(self.cell_num):
            for j in range(5):
                pass