document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM content loaded, fetching recommendations...');
    fetchRecommendations();
});

async function fetchRecommendations() {
    try {
        console.log('Fetching recommendations from API...');
        const response = await fetch('/api/stock-recommendations');
        const result = await response.json();
        console.log('Received data:', result);
        
        if (result.status === 'success') {
            console.log('Successfully received data, updating UI...');
            updateUI(result.data);
        } else {
            console.error('Error from server:', result.message);
        }
    } catch (error) {
        console.error('Error fetching recommendations:', error);
    }
}

function updateUI(data) {
    console.log('Updating UI with data:', data);
    updateSection('day-recommendations', data['1_day']);
    updateSection('month-recommendations', data['1_month']);
    updateSection('year-recommendations', data['1_year']);
    updateSection('decade-recommendations', data['10_years']);
    console.log('UI update complete');
}

function updateSection(sectionId, recommendations) {
    console.log(`Updating section ${sectionId} with:`, recommendations);
    const section = document.getElementById(sectionId);
    if (!section) {
        console.error(`Section with id ${sectionId} not found`);
        return;
    }
    const list = section.querySelector('.stock-list');
    if (!list) {
        console.error(`Stock list not found in section ${sectionId}`);
        return;
    }
    list.innerHTML = '';

    if (!recommendations || recommendations.length === 0) {
        list.innerHTML = '<li>No recommendations available</li>';
        return;
    }

    recommendations.forEach(stock => {
        const li = document.createElement('li');
        if (stock.expected_return !== undefined) {
            li.innerHTML = `
                <span class="ticker">${stock.ticker}</span>
                <span class="metric">${(stock.expected_return * 100).toFixed(2)}%</span>
            `;
        } else {
            li.innerHTML = `
                <span class="ticker">${stock.ticker}</span>
                <span class="score">Score: ${stock.score.toFixed(2)}</span>
                <div class="details">
                    <p>P/E Ratio: ${stock.pe_ratio ? stock.pe_ratio.toFixed(2) : 'N/A'}</p>
                    <p>P/B Ratio: ${stock.pb_ratio ? stock.pb_ratio.toFixed(2) : 'N/A'}</p>
                    <p>Profit Margin: ${stock.profit_margin ? (stock.profit_margin * 100).toFixed(2) + '%' : 'N/A'}</p>
                    <p>ROE: ${stock.roe ? (stock.roe * 100).toFixed(2) + '%' : 'N/A'}</p>
                    <p>Dividend Yield: ${stock.dividend_yield ? (stock.dividend_yield * 100).toFixed(2) + '%' : 'N/A'}</p>
                    <p>5-Year CAGR: ${stock.cagr ? (stock.cagr * 100).toFixed(2) + '%' : 'N/A'}</p>
                    <p>Volatility: ${stock.volatility ? (stock.volatility * 100).toFixed(2) + '%' : 'N/A'}</p>
                </div>
            `;
        }
        list.appendChild(li);
    });
    console.log(`Section ${sectionId} update complete`);
}

fetchRecommendations();

setInterval(fetchRecommendations, 300000);
