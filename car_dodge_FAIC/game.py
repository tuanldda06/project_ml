import random
import pygame
import cv2
import numpy as np
from settings import *
from objects import Car
from renderer import draw_background, draw_player, draw_enemy, draw_overlay, draw_particles
from hand_input import HandTracker

class CarDodgeCVTasksGame:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("segoeui", 22, bold=True)
        self.font_big = pygame.font.SysFont("segoeui", 46, bold=True)
        self.tracker = HandTracker(CAM_W, CAM_H)
        self.cam_surface = None 
        self.btn_rect = pygame.Rect(SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2, 200, 60)
        
        # Âm lượng nhỏ
        self.sound_score = self._create_beep_sound(880, 0.1, 0.05)
        self.sound_crash = self._create_crash_sound(0.4, 0.1)

        self.lanes = LANE_CENTERS[:]
        self.player_lane = 2
        self.player = Car(lane_index=2, x=self.lanes[2], y=PLAYER_Y, width=PLAYER_W, height=PLAYER_H)
        
        self.enemies, self.particles, self.score, self.best_score, self.lives = [], [], 0, 0, 3
        self.running, self.paused, self.game_over, self.keyboard_mode = True, True, False, False
        self.spawn_timer, self.enemy_speed, self.scroll_y = 0.0, 250.0, 0.0

    def _create_beep_sound(self, freq, dur, vol):
        sr = 44100
        t = np.linspace(0, dur, int(sr*dur), False)
        wave = np.sin(freq * t * 2 * np.pi) * np.exp(-t * 10) * vol
        audio = np.int16(wave * 32767)
        return pygame.sndarray.make_sound(np.column_stack((audio, audio)))

    def _create_crash_sound(self, dur, vol):
        sr = 44100
        wave = np.random.uniform(-1, 1, int(sr*dur)) * np.exp(-np.linspace(0, dur, int(sr*dur)) * 5) * vol
        audio = np.int16(wave * 32767)
        return pygame.sndarray.make_sound(np.column_stack((audio, audio)))

    def reset(self):
        self.player_lane, self.player.x, self.enemies, self.particles, self.score, self.lives = 2, self.lanes[2], [], [], 0, 3
        self.enemy_speed, self.paused, self.game_over = 250.0, False, False

    def handle_events(self, hand_state):
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.btn_rect.collidepoint(event.pos):
                    if self.game_over: self.reset()
                    else: self.paused = not self.paused
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m: self.keyboard_mode = not self.keyboard_mode
                if event.key == pygame.K_ESCAPE: self.running = False
        
        if hand_state.found and not self.keyboard_mode:
            x_val = max(0.0, min(1.0, (hand_state.x_norm - 0.2) / 0.6))
            self.player_lane = min(LANE_COUNT - 1, int(x_val * LANE_COUNT))
        self.player.x += (self.lanes[self.player_lane] - self.player.x) * 0.3
        return mouse_pos

    def update(self, dt):
        if self.paused or self.game_over: return
        self.enemy_speed = 250 + min(500, self.score * 5)
        self.scroll_y += self.enemy_speed * 1.5 * dt
        
        if random.random() > 0.5:
            self.particles.append([self.player.x - 15, self.player.y + 80, random.randint(4, 8), 150])
            self.particles.append([self.player.x + 15, self.player.y + 80, random.randint(4, 8), 150])
        for p in self.particles:
            p[1] += self.enemy_speed * dt; p[3] -= 5
        self.particles = [p for p in self.particles if p[3] > 0]

        self.spawn_timer += dt
        if self.spawn_timer >= max(0.4, 2.0 - self.score * 0.02):
            self.spawn_timer = 0.0
            lane = random.randint(0, LANE_COUNT-1)
            self.enemies.append(Car(lane_index=lane, x=self.lanes[lane], y=-ENEMY_H, width=ENEMY_W, height=ENEMY_H, color=(random.randint(50,255), 50, 50), speed=self.enemy_speed))

        for e in self.enemies:
            e.y += e.speed * dt
            if e.rect().inflate(-20, -20).colliderect(self.player.rect()):
                self.sound_crash.play(); self.lives -= 1; e.y = 2000
                if self.lives <= 0: self.game_over = True; self.paused = True; self.best_score = max(self.best_score, self.score)
            elif e.y > self.player.y and not e.passed:
                self.sound_score.play(); e.passed = True; self.score += 1
        self.enemies = [e for e in self.enemies if e.y < SCREEN_HEIGHT + 100]

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            hand_state, frame = self.tracker.read()
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.cam_surface = pygame.image.frombuffer(cv2.resize(rgb, (CAM_W, CAM_H)).tobytes(), (CAM_W, CAM_H), "RGB")
            
            mouse_pos = self.handle_events(hand_state)
            self.update(dt)
            
            draw_background(self.screen, self.font_small, self.font_big, self.score, self.best_score, self.lives, "KEYBOARD" if self.keyboard_mode else "HAND", self.player_lane, hand_state.found, self.clock.get_fps(), self.scroll_y, self.enemy_speed)
            draw_particles(self.screen, self.particles)
            for e in self.enemies: draw_enemy(self.screen, e)
            draw_player(self.screen, self.player)
            
            # Camera to và căn lề
            if self.cam_surface: 
                cam_x = SCREEN_WIDTH - SIDEBAR_W + 10
                cam_y = 24
                pygame.draw.rect(self.screen, ACCENT, (cam_x - 2, cam_y - 2, CAM_W + 4, CAM_H + 4), border_radius=10)
                self.screen.blit(self.cam_surface, (cam_x, cam_y))
                
            if self.paused and not self.game_over:
                draw_overlay(self.screen, self.font_big, self.font_small, "CAR DODGE", "START GAME", self.btn_rect, mouse_pos)
            elif self.game_over:
                draw_overlay(self.screen, self.font_big, self.font_small, "GAME OVER", "RETRY", self.btn_rect, mouse_pos)
            
            pygame.display.flip()
        self.tracker.close(); pygame.quit()