import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # Oculta "pygame-ce ..."

import sys
import ctypes
import pygame

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SDH.Hubble.2.1.0")  # Cria AppID para Windows
sys.argv[0] = "Hubble"

print("SDH [2.1.0], Hello Hubble")

# --- INICIALIZAÇÃO DO PYGAME ---
pygame.init()
VERSION = "SDH [2.1.0], Hello Hubble"

WIDTH, HEIGHT = 800, 600
_screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(VERSION)

# --- ÍCONE PERSONALIZADO ---
try:
    icon = pygame.image.load("Hubble_ico.png")
    pygame.display.set_icon(icon)
except Exception as e:
    print(f"[SDH] Aviso: ícone não encontrado ({e})")
    pygame.display.set_icon(pygame.Surface((1, 1)))

# --- CORES ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# --- FUNÇÕES PRINCIPAIS ---

def init(width=800, height=600, title=VERSION):
    """Inicializa a janela do SDH"""
    global _screen, WIDTH, HEIGHT
    WIDTH, HEIGHT = width, height
    _screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(title)

    try:
        icon = pygame.image.load("Hubble_ico.png")
        pygame.display.set_icon(icon)
    except:
        pygame.display.set_icon(pygame.Surface((1, 1)))

def clear(color=(0, 0, 0)):
    _screen.fill(color)

def draw_circle(x, y, radius, color=(255, 255, 255)):
    pygame.draw.circle(_screen, color, (int(x), int(y)), int(radius))

def draw_rect(x, y, w, h, color=(255, 255, 255)):
    pygame.draw.rect(_screen, color, (int(x), int(y), int(w), int(h)))

def draw_text(text, x, y, size=20, color=(255, 255, 255)):
    font = pygame.font.SysFont("consolas", size)
    txt = font.render(str(text), True, color)
    _screen.blit(txt, (int(x), int(y)))

def update():
    pygame.display.flip()

# --- EVENTOS / TECLADO ---
_pressed_keys = set()

def handle_events():
    global _pressed_keys
    evs = []
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            evs.append({"type": "quit"})
        elif e.type == pygame.KEYDOWN:
            _pressed_keys.add(e.key)
            evs.append({"type": "key_down", "key": e.key})
        elif e.type == pygame.KEYUP:
            if e.key in _pressed_keys:
                _pressed_keys.remove(e.key)
            evs.append({"type": "key_up", "key": e.key})
    return evs

def key_pressed(key):
    return key in _pressed_keys

# --- ÁUDIO / SONS ---
pygame.mixer.init()

def load_sound(path):
    try:
        return pygame.mixer.Sound(path)
    except Exception as e:
        print(f"[SDH] Erro ao carregar som '{path}': {e}")
        return None

def play_sound(sound, loops=0):
    if sound:
        sound.play(loops=loops)

def stop_all_sounds():
    pygame.mixer.stop()

# --- ENCERRAR ---
def quit():
    pygame.quit()
    sys.exit()