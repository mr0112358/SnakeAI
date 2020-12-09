from typing import List, Tuple, Union, Dict


class Slope(object):
    __slots__ = ('rise', 'run')
    def __init__(self, rise: int, run: int):
        self.rise = rise
        self.run = run

class Point(object):
    __slots__ = ('x', 'y')
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def copy(self) -> 'Point':
        x = self.x
        y = self.y
        return Point(x, y)

    def to_dict(self) -> Dict[str, int]:
        d = {}
        d['x'] = self.x
        d['y'] = self.y
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> 'Point':
        return Point(d['x'], d['y'])

    def __eq__(self, other: Union['Point', Tuple[int, int]]) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return other[0] == self.x and other[1] == self.y
        elif isinstance(other, Point) and self.x == other.x and self.y == other.y:
            return True
        return False

    def __sub__(self, other: Union['Point', Tuple[int, int]]) -> 'Point':
        if isinstance(other, tuple) and len(other) == 2:
            diff_x = self.x - other[0]
            diff_y = self.y - other[1]
            return Point(diff_x, diff_y)
        elif isinstance(other, Point):
            diff_x = self.x - other.x
            diff_y = self.y - other.y
            return Point(diff_x, diff_y)
        return None

    def __rsub__(self, other: Tuple[int, int]):
        diff_x = other[0] - self.x
        diff_y = other[1] - self.y
        return Point(diff_x, diff_y)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __str__(self) -> str:
        return '({}, {})'.format(self.x, self.y)


### These lines are defined such that facing "up" would be L0 ###
# Create 16 lines to be able to "see" around
VISION_16 = (
#   L0            L1             L2             L3
    Slope(-1, 0), Slope(-2, 1),  Slope(-1, 1),  Slope(-1, 2),
#   L4            L5             L6             L7      
    Slope(0, 1),  Slope(1, 2),   Slope(1, 1),   Slope(2, 1),
#   L8            L9             L10            L11
    Slope(1, 0),  Slope(2, -1),  Slope(1, -1),  Slope(1, -2),
#   L12           L13            L14            L15
    Slope(0, -1), Slope(-1, -2), Slope(-1, -1), Slope(-2, -1)
)

# Create 8 lines to be able to "see" around
# Really just VISION_16 without odd numbered lines

# VISION_8
# if self.apple_location.x - self.snake_array[0].x <= 0 and self.apple_location.y - self.snake_array[0].y > 0 and self.direction == 'u':
# output mirror = False rotate = 0
# VISION_8 = tuple([VISION_16[i] for i in range(len(VISION_16)) if i%2==0])
VISION_8 = (
#   L0            L1
    Slope(-1, 0),  Slope(-1, 1),
#   L2            L3
    Slope(0, 1),  Slope(1, 1),
#   L4            L5
    Slope(1, 0), Slope(1, -1),
#   L6            L7
    Slope(0, -1), Slope(-1, -1)

)

# Create 4 lines to be able to "see" around
# Really just VISION_16 but removing anything not divisible by 4
VISION_4 = tuple([VISION_16[i] for i in range(len(VISION_16)) if i%4==0])

# Transpose VISION_8 90 degrees
# if self.apple_location.x - self.snake_array[0].x < 0 and self.apple_location.y - self.snake_array[0].y <= 0 and self.direction == 'l':
# output mirror = False rotate = 90
VISION_8_t90 = (
#   L0            L1
    Slope(0, -1),  Slope(-1, -1),
#   L2            L3
    Slope(-1, 0),  Slope(-1, 1),
#   L4            L5
    Slope(0, 1), Slope(1, 1),
#   L6            L7
    Slope(1, 0), Slope(1, -1)

)


# Transpose VISION_8 180 degrees
# if self.apple_location.x - self.snake_array[0].x >= 0 and self.apple_location.y - self.snake_array[0].y < 0 and self.direction == 'd':
# output mirror = False rotate = 180
VISION_8_t180 = (
#   L0            L1
    Slope(1, 0),  Slope(1, -1),
#   L2            L3
    Slope(0, -1),  Slope(-1, -1),
#   L4            L5
    Slope(-1, 0), Slope(-1, 1),
#   L6            L7
    Slope(0, 1), Slope(1, 1)

) 

# Transpose VISION_8 270 degrees
# if self.apple_location.x - self.snake_array[0].x > 0 and self.apple_location.y - self.snake_array[0].y >= 0 and self.direction == 'r':
# output mirror = False rotate = 270
VISION_8_t270 = (
#   L0            L1
    Slope(0, 1),  Slope(1, 1),
#   L2            L3
    Slope(1, 0),  Slope(1, -1),
#   L4            L5
    Slope(0, -1), Slope(-1, -1),
#   L6            L7
    Slope(-1, 0), Slope(-1, 1)

) 

# Mirror VISION_8
# if self.apple_location.x - self.snake_array[0].x >= 0 and self.apple_location.y - self.snake_array[0].y > 0 and self.direction == 'u':
# output mirror = True rotate = 0
VISION_8_m = (
#   L0            L1
    Slope(1, 0),  Slope(1, 1),
#   L2            L3
    Slope(0, 1),  Slope(-1, 1),
#   L4            L5
    Slope(-1, 0), Slope(-1, -1),
#   L6            L7 
    Slope(0, -1), Slope(1, -1)

)

# Mirror VISION_8_t90
# if self.apple_location.x - self.snake_array[0].x > 0 and self.apple_location.y - self.snake_array[0].y <= 0 and self.direction == 'r':
# output mirror = True rotate = 90
VISION_8_t90_m = (
#   L0            L1
    Slope(0, 1), Slope(-1, 1),
#   L2            L3
    Slope(-1, 0),  Slope(-1, -1),
#   L4            L5
    Slope(0, -1),  Slope(1, -1),
#   L6            L7 
    Slope(1, 0), Slope(1, 1)

)

# Mirror VISION_8_t180
# if self.apple_location.x - self.snake_array[0].x <= 0 and self.apple_location.y - self.snake_array[0].y < 0 and self.direction == 'd':
# output mirror = True rotate = 180
VISION_8_t180_m = (
#   L0            L1
    Slope(-1, 0), Slope(-1, -1),
#   L2            L3
    Slope(0, -1), Slope(1, -1),
#   L4            L5
    Slope(1, 0),  Slope(1, 1),
#   L6            L7 
    Slope(0, 1),  Slope(-1, 1)

)

# Mirror VISION_8_t270
# if self.apple_location.x - self.snake_array[0].x < 0 and self.apple_location.y - self.snake_array[0].y >= 0 and self.direction == 'l':
# output mirror = True rotate = 270
VISION_8_t270_m = (
#   L0            L1
    Slope(0, 1),  Slope(-1, 1),
#   L2            L3
    Slope(-1, 0),  Slope(-1, -1),
#   L4            L5
    Slope(0, -1),  Slope(1, -1),
#   L6            L7 
    Slope(1, 0), Slope(1, 1)

)

# Direction downwards
# if self.apple_location.x - self.snake_array[0].x <= 0 and self.apple_location.y - self.snake_array[0].y > 0 and self.direction == 'd':
# output mirror = False rotate = 0
VISION_8_d = (
#   L0            L1
    Slope(-1, 0),  Slope(-1, -1),
#   L2            L3
    Slope(0, -1),  Slope(1, -1),
#   L4            L5
    Slope(1, 0), Slope(1, 1),
#   L6            L7 
    Slope(0, 1), Slope(-1, 1)

)

# Direction downwards transpose 90
# if self.apple_location.x - self.snake_array[0].x < 0 and self.apple_location.y - self.snake_array[0].y <= 0 and self.direction == 'r':
# output mirror = False rotate = 90
VISION_8_d_t90 = (
#   L0            L1
    Slope(0, 1),  Slope(-1, 1),
#   L2            L3
    Slope(-1, 0),  Slope(-1, -1),
#   L4            L5
    Slope(0, -1), Slope(1, -1),
#   L6            L7
    Slope(1, 0), Slope(1, 1)

) 

# Direction downwards transpose 180
# if self.apple_location.x - self.snake_array[0].x >= 0 and self.apple_location.y - self.snake_array[0].y < 0 and self.direction == 'u':
# output mirror = False rotate = 180
VISION_8_d_t180 = (
#   L0            L1
    Slope(1, 0),  Slope(1, 1),
#   L2            L3
    Slope(0, 1),  Slope(-1, 1),
#   L4            L5
    Slope(-1, 0), Slope(-1, -1),
#   L6            L7
    Slope(0, -1), Slope(1, -1)

) 

# Direction downwards transpose 270
# if self.apple_location.x - self.snake_array[0].x > 0 and self.apple_location.y - self.snake_array[0].y >= 0 and self.direction == 'l':
# output mirror = False rotate = 270
VISION_8_d_t270 = (
#   L0            L1
    Slope(0, -1),  Slope(1, -1),
#   L2            L3
    Slope(1, 0),  Slope(1, 1),
#   L4            L5
    Slope(0, 1), Slope(-1, 1),
#   L6            L7
    Slope(-1, 0), Slope(-1, -1)

) 

# Mirror Direction downwards
# if self.apple_location.x - self.snake_array[0].x >= 0 and self.apple_location.y - self.snake_array[0].y > 0 and self.direction == 'd':
# output mirror = True rotate = 0
VISION_8_d_m = (
#   L0            L1
    Slope(1, 0),  Slope(1, -1),
#   L2            L3
    Slope(0, -1),  Slope(-1, -1),
#   L4            L5
    Slope(-1, 0), Slope(-1, 1),
#   L6            L7 
    Slope(0, 1), Slope(1, 1)

)

# Mirror Direction downwards transpose 90
# if self.apple_location.x - self.snake_array[0].x > 0 and self.apple_location.y - self.snake_array[0].y <= 0 and self.direction == 'l':
# output mirror = True rotate = 90
VISION_8_d_t90_m = (
#   L0            L1
    Slope(0, -1),  Slope(-1, -1),
#   L2            L3
    Slope(-1, 0),  Slope(-1, 1),
#   L4            L5
    Slope(0, 1), Slope(1, 1),
#   L6            L7
    Slope(1, 0), Slope(1, -1)

) 

# Mirror Direction downwards transpose 180
# if self.apple_location.x - self.snake_array[0].x <= 0 and self.apple_location.y - self.snake_array[0].y < 0 and self.direction == 'u':
# output mirror = True rotate = 180
VISION_8_d_t180_m = (
#   L0            L1
    Slope(-1, 0),  Slope(-1, 1),
#   L2            L3
    Slope(0, 1),  Slope(1, 1),
#   L4            L5
    Slope(1, 0), Slope(1, -1),
#   L6            L7
    Slope(0, -1), Slope(-1, -1)

) 

# Mirror Direction downwards transpose 270
# if self.apple_location.x - self.snake_array[0].x < 0 and self.apple_location.y - self.snake_array[0].y >= 0 and self.direction == 'r':
# output mirror = True rotate = 270
VISION_8_d_t270_m = (
#   L0            L1
    Slope(0, 1),  Slope(1, 1),
#   L2            L3
    Slope(1, 0),  Slope(1, -1),
#   L4            L5
    Slope(0, -1), Slope(-1, -1),
#   L6            L7
    Slope(-1, 0), Slope(-1, 1)

) 

        