/*
================================================================================
 * Sentiment Analysis Dataset - JavaScript
 * 
 * Project: Sentiment Analysis Dataset
 * Description: Text sentiment analysis dataset with labeled reviews, comments,
 *              and social media posts for sentiment classification models.
 * Category: Text Data
 * Difficulty: Intermediate
 * 
 * Author: Molla Samser (Founder)
 * Designer & Tester: Rima Khatun
 * Website: https://rskworld.in
 * Email: help@rskworld.in | support@rskworld.in
 * Phone: +91 93305 39277
 * 
 * © 2026 RSK World - Free Programming Resources & Source Code
 * All rights reserved.
================================================================================
*/

// Sample Data for Explorer
const sampleData = [
    {
        id: 1,
        text: "Absolutely love this product! It exceeded all my expectations and the quality is outstanding. Will definitely recommend to friends and family.",
        sentiment: "positive",
        source: "Product Review",
        icon: "fas fa-shopping-cart"
    },
    {
        id: 2,
        text: "The service was okay, nothing special. Delivery was on time but the packaging could have been better. Average experience overall.",
        sentiment: "neutral",
        source: "Customer Feedback",
        icon: "fas fa-comment"
    },
    {
        id: 3,
        text: "Terrible experience! The product arrived damaged and customer support was unhelpful. Complete waste of money. Never ordering again.",
        sentiment: "negative",
        source: "Product Review",
        icon: "fas fa-shopping-cart"
    },
    {
        id: 4,
        text: "This is the best purchase I've made this year! The features are amazing and it works exactly as advertised. Highly satisfied customer here!",
        sentiment: "positive",
        source: "Social Media",
        icon: "fab fa-twitter"
    },
    {
        id: 5,
        text: "Received the package today. It looks decent but haven't tried it yet. Will update after a week of use.",
        sentiment: "neutral",
        source: "Social Media",
        icon: "fab fa-facebook"
    },
    {
        id: 6,
        text: "So disappointed with this purchase. The description was misleading and the actual product is nothing like what was shown in the pictures.",
        sentiment: "negative",
        source: "Product Review",
        icon: "fas fa-shopping-cart"
    },
    {
        id: 7,
        text: "Great value for money! The build quality is solid and it performs really well. Shipping was fast too. Very happy with this purchase!",
        sentiment: "positive",
        source: "Customer Feedback",
        icon: "fas fa-comment"
    },
    {
        id: 8,
        text: "Just unboxed my order. Standard quality, meets basic requirements. Not exceptional but does the job. Fair price for what you get.",
        sentiment: "neutral",
        source: "Social Media",
        icon: "fab fa-instagram"
    },
    {
        id: 9,
        text: "Worst customer service I've ever experienced. Three weeks and my issue is still not resolved. Absolutely frustrating and unprofessional.",
        sentiment: "negative",
        source: "Customer Feedback",
        icon: "fas fa-comment"
    },
    {
        id: 10,
        text: "Five stars! This product changed my daily routine for the better. Excellent quality, beautiful design, and amazing functionality!",
        sentiment: "positive",
        source: "Product Review",
        icon: "fas fa-shopping-cart"
    },
    {
        id: 11,
        text: "The product works as expected. Nothing extraordinary but nothing to complain about either. Average performance for the price point.",
        sentiment: "neutral",
        source: "Product Review",
        icon: "fas fa-shopping-cart"
    },
    {
        id: 12,
        text: "Completely broken on arrival! Requested a refund but still waiting after two weeks. This company has lost a customer forever.",
        sentiment: "negative",
        source: "Social Media",
        icon: "fab fa-twitter"
    },
    {
        id: 13,
        text: "Impressed beyond words! The attention to detail is remarkable. Best investment I've made in a long time. Truly exceptional product!",
        sentiment: "positive",
        source: "Customer Feedback",
        icon: "fas fa-comment"
    },
    {
        id: 14,
        text: "Received my order yesterday. First impressions are neither good nor bad. Will need more time to form a complete opinion.",
        sentiment: "neutral",
        source: "Social Media",
        icon: "fab fa-facebook"
    },
    {
        id: 15,
        text: "Don't waste your money on this garbage! Poor quality materials, horrible design, and falls apart after first use. Total scam!",
        sentiment: "negative",
        source: "Product Review",
        icon: "fas fa-shopping-cart"
    }
];

// Additional samples for loading more
const additionalSamples = [
    {
        id: 16,
        text: "This exceeded all my expectations! Fast shipping, beautiful packaging, and the product itself is top-notch. Definitely buying more!",
        sentiment: "positive",
        source: "Product Review",
        icon: "fas fa-shopping-cart"
    },
    {
        id: 17,
        text: "It's an okay product. Functions as described but nothing special about it. Wouldn't go out of my way to recommend it.",
        sentiment: "neutral",
        source: "Customer Feedback",
        icon: "fas fa-comment"
    },
    {
        id: 18,
        text: "Massive letdown! The photos made it look premium but in reality it's cheap plastic. False advertising at its finest.",
        sentiment: "negative",
        source: "Social Media",
        icon: "fab fa-instagram"
    }
];

// Global Variables
let currentFilter = 'all';
let displayedSamples = 9;
let allSamples = [...sampleData];

// DOM Elements
const samplesGrid = document.getElementById('samplesGrid');
const filterTabs = document.querySelectorAll('.filter-tab');
const loadMoreBtn = document.getElementById('loadMore');
const backToTopBtn = document.getElementById('backToTop');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    renderSamples();
    initializeCharts();
    initializeCounters();
    setupEventListeners();
});

// Render Sample Cards
function renderSamples() {
    const filteredSamples = currentFilter === 'all' 
        ? allSamples 
        : allSamples.filter(sample => sample.sentiment === currentFilter);
    
    const samplesToShow = filteredSamples.slice(0, displayedSamples);
    
    samplesGrid.innerHTML = samplesToShow.map((sample, index) => `
        <div class="sample-card" style="animation-delay: ${index * 0.1}s" data-sentiment="${sample.sentiment}">
            <div class="sample-header">
                <span class="sample-source">
                    <i class="${sample.icon}"></i>
                    ${sample.source}
                </span>
                <span class="sample-sentiment ${sample.sentiment}">
                    <i class="fas fa-${getSentimentIcon(sample.sentiment)}"></i>
                    ${capitalizeFirst(sample.sentiment)}
                </span>
            </div>
            <p class="sample-text">"${sample.text}"</p>
        </div>
    `).join('');
    
    // Show/hide load more button
    loadMoreBtn.style.display = samplesToShow.length < filteredSamples.length ? 'inline-flex' : 'none';
}

// Get Sentiment Icon
function getSentimentIcon(sentiment) {
    switch(sentiment) {
        case 'positive': return 'smile';
        case 'neutral': return 'meh';
        case 'negative': return 'frown';
        default: return 'circle';
    }
}

// Capitalize First Letter
function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

// Setup Event Listeners
function setupEventListeners() {
    // Filter Tabs
    filterTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            filterTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentFilter = tab.dataset.filter;
            displayedSamples = 9;
            renderSamples();
        });
    });
    
    // Load More Button
    loadMoreBtn.addEventListener('click', () => {
        displayedSamples += 6;
        
        // Add additional samples if needed
        if (displayedSamples > allSamples.length && additionalSamples.length > 0) {
            allSamples = [...allSamples, ...additionalSamples];
        }
        
        renderSamples();
    });
    
    // Back to Top Button
    window.addEventListener('scroll', () => {
        if (window.scrollY > 500) {
            backToTopBtn.classList.add('visible');
        } else {
            backToTopBtn.classList.remove('visible');
        }
    });
    
    backToTopBtn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    
    // Mobile Menu Toggle
    const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');
    
    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', () => {
            navLinks.classList.toggle('active');
        });
    }
}

// Initialize Charts
function initializeCharts() {
    // Sentiment Distribution Chart
    const sentimentCtx = document.getElementById('sentimentChart');
    if (sentimentCtx) {
        new Chart(sentimentCtx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [37, 30.4, 32.6],
                    backgroundColor: [
                        '#28a745',
                        '#ffc107',
                        '#dc3545'
                    ],
                    borderWidth: 0,
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#b3b3b3',
                            padding: 15,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.raw}%`;
                            }
                        }
                    }
                },
                cutout: '65%'
            }
        });
    }
    
    // Data Sources Chart
    const sourceCtx = document.getElementById('sourceChart');
    if (sourceCtx) {
        new Chart(sourceCtx, {
            type: 'bar',
            data: {
                labels: ['Reviews', 'Social Media', 'Comments'],
                datasets: [{
                    label: 'Samples',
                    data: [22000, 18000, 10000],
                    backgroundColor: [
                        'rgba(220, 53, 69, 0.8)',
                        'rgba(220, 53, 69, 0.6)',
                        'rgba(220, 53, 69, 0.4)'
                    ],
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#b3b3b3',
                            callback: function(value) {
                                return value / 1000 + 'K';
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#b3b3b3'
                        }
                    }
                }
            }
        });
    }
}

// Initialize Counter Animation
function initializeCounters() {
    const counters = document.querySelectorAll('.metric-value[data-count]');
    
    const observerOptions = {
        threshold: 0.5
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const counter = entry.target;
                const target = parseInt(counter.dataset.count);
                animateCounter(counter, target);
                observer.unobserve(counter);
            }
        });
    }, observerOptions);
    
    counters.forEach(counter => observer.observe(counter));
}

// Animate Counter
function animateCounter(element, target) {
    const duration = 2000;
    const start = 0;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smooth animation
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        
        const current = Math.floor(start + (target - start) * easeOutQuart);
        element.textContent = current.toLocaleString();
        
        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.textContent = target.toLocaleString();
        }
    }
    
    requestAnimationFrame(update);
}

// Smooth Scroll for Anchor Links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add animation on scroll for cards
const animateOnScroll = () => {
    const cards = document.querySelectorAll('.feature-card, .case-card, .tech-card, .sample-card');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });
    
    cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'all 0.5s ease';
        observer.observe(card);
    });
};

// Call animation on load
window.addEventListener('load', animateOnScroll);

