import numpy as np
from sys import exit

class Boltzmann(object):
    def __init__(self, width, theta_hh, theta_hx):
        self.width = width
        self.theta_hh = theta_hh
        self.theta_hx = theta_hx

    def get_numerator(self, pi_old, noisy_image, (x,y)):
        num_exp_sum = 0.
        if(x > 0):
            num_exp_sum += (self.theta_hh * (2.*pi_old[x-1,y]-1.) + self.theta_hx * noisy_image[x-1,y])
        if(x < self.width-1):
            num_exp_sum += (self.theta_hh * (2.*pi_old[x+1,y]-1.) + self.theta_hx * noisy_image[x+1,y])
        if(y > 0):
            num_exp_sum += (self.theta_hh * (2.*pi_old[x,y-1]-1.) + self.theta_hx * noisy_image[x,y-1])
        if(y < self.width-1):
            num_exp_sum += (self.theta_hh * (2.*pi_old[x,y+1]-1.) + self.theta_hx * noisy_image[x,y+1])
        return np.exp(num_exp_sum)

    def get_denomenator(self, pi_old, noisy_image, (x,y)):
        den_exp_sum_left = 0.
        if(x > 0):
            den_exp_sum_left += (self.theta_hh * (2.*pi_old[x-1,y]-1.) + self.theta_hx * noisy_image[x-1,y])
        if(x < self.width-1):
            den_exp_sum_left += (self.theta_hh * (2.*pi_old[x+1,y]-1.) + self.theta_hx * noisy_image[x+1,y])
        if(y > 0):
            den_exp_sum_left += (self.theta_hh * (2.*pi_old[x,y-1]-1.) + self.theta_hx * noisy_image[x,y-1])
        if(y < self.width-1):
            den_exp_sum_left += (self.theta_hh * (2.*pi_old[x,y+1]-1.) + self.theta_hx * noisy_image[x,y+1])

        den_exp_sum_right = 0.
        if(x > 0):
            den_exp_sum_right += -1. * (self.theta_hh * (2.*pi_old[x-1,y]-1.) + self.theta_hx * noisy_image[x-1,y])
        if(x < self.width-1):
            den_exp_sum_right += -1. * (self.theta_hh * (2.*pi_old[x+1,y]-1.) + self.theta_hx * noisy_image[x+1,y])
        if(y > 0):
            den_exp_sum_right += -1. * (self.theta_hh * (2.*pi_old[x,y-1]-1.) + self.theta_hx * noisy_image[x,y-1])
        if(y < self.width-1):
            den_exp_sum_right += -1. * (self.theta_hh * (2.*pi_old[x,y+1]-1.) + self.theta_hx * noisy_image[x,y+1])
        return np.exp(den_exp_sum_left) + np.exp(den_exp_sum_right)

    def get_pi_new(self, pi_old, image):
        pi_new = np.zeros((self.width, self.width))
        for x in range(self.width):
            for y in range(self.width):
                numerator = self.get_numerator(pi_old, image, (x,y))
                denominator = self.get_denomenator(pi_old, image, (x,y))
                pi_new[x,y]= numerator/float(denominator)
        return pi_new

    @staticmethod
    def get_diff(old, new):
        return np.sum(np.absolute(np.subtract(old, new)))

    def denoise_image(self, image):
        pi_old = np.ones((self.width, self.width))/5.
        prev_diff = float('Inf')
        curr_diff = 0
        while True:
            pi_new = self.get_pi_new(pi_old, image)
            curr_diff = Boltzmann.get_diff(pi_old, pi_new)
            if abs(curr_diff - prev_diff) < 0.1:
                break
            pi_old = pi_new
            prev_diff = curr_diff
        new_image = (pi_new > 0.2) * 2 - 1
        return new_image
