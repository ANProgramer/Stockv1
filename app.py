from flask import Flask, jsonify, render_template
from stock_analysis.analysis import get_stock_recommendations
from datetime import datetime  
import logging

app = Flask(__name__, static_url_path='/static')
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stock-recommendations')
def stock_recommendations():
    try:
        logging.info("Fetching stock recommendations")
        recommendations = get_stock_recommendations()
        logging.info("Successfully fetched stock recommendations")
        return jsonify({
            'status': 'success',
            'data': recommendations,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logging.error(f"Error in stock_recommendations: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    app.run(debug=True)