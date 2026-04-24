import pygame
from settings import *

def draw_button(screen, font, text, x, y, w, h, is_hovered):
    color = ACCENT if is_hovered else (45, 55, 75)
    pygame.draw.rect(screen, color, (x, y, w, h), border_radius=10)
    pygame.draw.rect(screen, (255, 255, 255), (x, y, w, h), width=2, border_radius=10)
    text_surf = font.render(text, True, TEXT)
    text_rect = text_surf.get_rect(center=(x + w//2, y + h//2))
    screen.blit(text_surf, text_rect)

def draw_dashboard_card(screen, font, title, value, x, y, w, h, val_color):
    pygame.draw.rect(screen, (35, 42, 58), (x, y, w, h), border_radius=10)
    screen.blit(font.render(title, True, (150, 160, 180)), (x + 12, y + 8))
    screen.blit(font.render(str(value), True, val_color), (x + 12, y + 34))

def draw_lives(screen, font, lives, x, y, w, h):
    pygame.draw.rect(screen, (35, 42, 58), (x, y, w, h), border_radius=10)
    screen.blit(font.render("Lives", True, (150, 160, 180)), (x + 12, y + 8))
    for i in range(3):
        color = GOOD if i < lives else (60, 70, 90)
        pygame.draw.rect(screen, color, (x + 12 + i * 28, y + 40, 20, 8), border_radius=4)

def draw_background(screen, font_small, font_big, score, best_score, lives, mode_text, lane_index, hand_found, fps, scroll_y, speed):
    screen.fill(BG_COLOR)
    pygame.draw.rect(screen, ROAD_COLOR, (ROAD_X, 0, ROAD_WIDTH, SCREEN_HEIGHT))
    pygame.draw.rect(screen, ROAD_EDGE, (ROAD_X, 0, 4, SCREEN_HEIGHT))
    pygame.draw.rect(screen, ROAD_EDGE, (ROAD_X + ROAD_WIDTH - 4, 0, 4, SCREEN_HEIGHT))

    for i in range(1, LANE_COUNT):
        x = ROAD_X + i * LANE_WIDTH
        y = (scroll_y % 80) - 80
        while y < SCREEN_HEIGHT:
            pygame.draw.rect(screen, LANE_MARK, (x - 3, y, 6, 40), border_radius=3)
            y += 80

    pygame.draw.rect(screen, (22, 28, 40), (0, 0, SIDEBAR_W, SCREEN_HEIGHT))
    pygame.draw.rect(screen, (22, 28, 40), (SCREEN_WIDTH - SIDEBAR_W, 0, SIDEBAR_W, SCREEN_HEIGHT))
    screen.blit(font_big.render("CAR DODGE", True, TEXT), (24, 20))
    pygame.draw.line(screen, ACCENT, (24, 75), (SIDEBAR_W - 24, 75), 3)

    # Các ô thông tin nhỏ gọn hơn
    draw_dashboard_card(screen, font_small, "Score", score, 24, 95, 150, 65, ACCENT)
    draw_dashboard_card(screen, font_small, "Best", best_score, 185, 95, 150, 65, GOOD)
    draw_lives(screen, font_small, lives, 24, 175, 150, 65)
    draw_dashboard_card(screen, font_small, "Camera", "DETECTED" if hand_found else "LOST", 185, 175, 150, 65, GOOD if hand_found else BAD)

def draw_particles(screen, particles):
    for p in particles:
        surface = pygame.Surface((p[2]*2, p[2]*2), pygame.SRCALPHA)
        pygame.draw.circle(surface, (200, 200, 200, int(p[3])), (p[2], p[2]), p[2])
        screen.blit(surface, (int(p[0] - p[2]), int(p[1] - p[2])))

def draw_car(screen, car, is_player=True):
    r = car.rect()
    main_color = (40, 120, 220) if is_player else car.color
    roof_color = (30, 90, 180) if is_player else (max(0, car.color[0]-40), max(0, car.color[1]-40), max(0, car.color[2]-40))
    bw, bh = 10, 28
    pygame.draw.rect(screen, (10, 10, 10), (r.x - 4, r.y + 25, bw, bh), border_radius=4)
    pygame.draw.rect(screen, (10, 10, 10), (r.right - 6, r.y + 25, bw, bh), border_radius=4)
    pygame.draw.rect(screen, (10, 10, 10), (r.x - 4, r.bottom - 50, bw, bh), border_radius=4)
    pygame.draw.rect(screen, (10, 10, 10), (r.right - 6, r.bottom - 50, bw, bh), border_radius=4)
    pygame.draw.rect(screen, main_color, r, border_radius=12)
    pygame.draw.rect(screen, (255, 255, 255) if is_player else (200, 200, 200), r, width=2, border_radius=12)
    pygame.draw.rect(screen, roof_color, (r.x + 12, r.y + 35, r.w - 24, r.h - 70), border_radius=6)
    pygame.draw.rect(screen, (20, 25, 35), (r.x + 14, r.y + 20, r.w - 28, 25), border_radius=4)
    pygame.draw.rect(screen, (20, 25, 35), (r.x + 14, r.bottom - 45, r.w - 28, 20), border_radius=4)

def draw_player(screen, car): draw_car(screen, car, is_player=True)
def draw_enemy(screen, car): draw_car(screen, car, is_player=False)

def draw_overlay(screen, font_big, font_small, title, btn_text, btn_rect, mouse_pos):
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill((10, 15, 25, 200))
    screen.blit(overlay, (0, 0))
    is_hovered = btn_rect.collidepoint(mouse_pos)
    draw_button(screen, font_small, btn_text, btn_rect.x, btn_rect.y, btn_rect.width, btn_rect.height, is_hovered)
    title_surf = font_big.render(title, True, TEXT)
    screen.blit(title_surf, title_surf.get_rect(center=(SCREEN_WIDTH//2, btn_rect.y - 60)))