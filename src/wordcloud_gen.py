import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

class WordShapePacker:
    def __init__(self, mask_path, words):
        print("WordShapePacker init")
        self.mask = self.load_mask(mask_path)
        self.words = words
        self.placed_rectangles = []
        self.min_font_size = 8
        self.max_font_size = 24
        self.fonts = {
            size: ImageFont.truetype("arial.ttf", size) 
            for size in range(self.min_font_size, self.max_font_size + 1, 2)
        }
        self.current_font_size = self.max_font_size
        
    def load_mask(self, mask_path):
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((600, 600))
        return np.array(mask) < 128
    
    def get_word_size(self, word, font_size):
        font = self.fonts[font_size]
        bbox = font.getbbox(word)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    def calculate_coverage(self):
        """Oblicza procent pokrycia maski"""
        if not self.placed_rectangles:
            return 0
            
        coverage = np.zeros_like(self.mask, dtype=bool)
        for x, y, w, h, _ in self.placed_rectangles:
            y_end = min(y + h, coverage.shape[0])
            x_end = min(x + w, coverage.shape[1])
            coverage[y:y_end, x:x_end] = True
            
        mask_area = np.sum(self.mask)
        covered_area = np.sum(coverage & self.mask)
        return covered_area / mask_area
    
    def can_place_rectangle(self, x, y, width, height):
        if (y + height > self.mask.shape[0] or 
            x + width > self.mask.shape[1] or 
            x < 0 or y < 0):
            return False
            
        steps_x = 4
        steps_y = 4
        for i in range(steps_y + 1):
            for j in range(steps_x + 1):
                px = x + (j * width) // steps_x
                py = y + (i * height) // steps_y
                if py >= self.mask.shape[0] or px >= self.mask.shape[1]:
                    return False # albo break, to dziala podobnie
                if not self.mask[py, px]:
                    return False
        
        margin = 1 
        for px, py, pw, ph, _ in self.placed_rectangles:
            if (x - margin < px + pw and x + width + margin > px and
                y - margin < py + ph and y + height + margin > py):
                return False
                
        return True
    
    def find_position(self, width, height, attempts):
        y_coords, x_coords = np.where(self.mask)
        if len(y_coords) == 0:
            return None
            
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))
        
        max_radius = max(self.mask.shape)
        step = 3  
        
        if self.can_place_rectangle(center_x - width//2, center_y - height//2, width, height):
            return center_x - width//2, center_y - height//2
        
        for radius in range(0, max_radius, step):
            angle_step = 0
            if attempts %300 == 0:
                angle_step += 1
            angles = np.linspace(0 + angle_step, 2*np.pi + angle_step, 32)  
            # random.shuffle(angles)  
            for angle in angles:
                x = int(center_x + radius * np.cos(angle) - width//2)
                y = int(center_y + radius * np.sin(angle) - height//2)
                
                if self.can_place_rectangle(x, y, width, height):
                    return x, y
        
        return None
    
    def pack_words(self):
        target_coverage = 0.85  
        placed_words = []
        attempts = 0
        max_attempts = 1000
        
        words_to_pack = []
        for size in self.fonts:  
            for word in self.words:
                words_to_pack.append((word, size))
                # words_to_pack.append((word, size))
        print(words_to_pack)
        words_to_pack.sort(key=lambda x: -x[1])  
        # random.shuffle(words_to_pack)
        
        while attempts < max_attempts:
            if not words_to_pack:
                sizes = sorted(self.fonts.keys())[:-2] 
                words_to_pack = [(word, size) for size in sizes for word in self.words] * 5 

            
            word, font_size = words_to_pack.pop(0)
            width, height = self.get_word_size(word, font_size)
            # width i heigth to wysokosc pierwszego słowa które jest w liście słów
            position = self.find_position(width, height, attempts)
            
            if position:
                x, y = position
                self.placed_rectangles.append((x, y, width, height, font_size))
                placed_words.append((word, x, y, font_size))
                
                coverage = self.calculate_coverage()
                print(coverage)
                if coverage >= target_coverage:
                    break
            
            attempts += 1
        print(attempts)
        return placed_words

    
    def visualize(self, output_path):
        img = Image.new('RGB', (self.mask.shape[1], self.mask.shape[0]), 'white')
        draw = ImageDraw.Draw(img)
        
        for word, x, y, font_size in self.pack_words():
            font = self.fonts[font_size]
            draw.text((x, y), word, fill='black', font=font)
            
        # img.save(output_path)
        return img

if __name__ == "__main__":
    words = ["Python", "Programming", "Algorithm", "Data", "Code", 
             "Computer", "Science", "Learning", "AI", "Development"]
    
    packer = WordShapePacker("src/cat_shape.png", words)
    packer.visualize("output.png")