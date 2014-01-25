package com.jetperch.touchlatency;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.view.MotionEvent;
import android.view.View;

public class TouchLatencyView extends View {
	private Path touchPath = new Path();
	private Paint touchPaint = new Paint();
	
	public TouchLatencyView(Context context) {
		super(context);
		
		touchPaint.setAntiAlias(true);
		touchPaint.setColor(Color.MAGENTA);
		touchPaint.setStrokeJoin(Paint.Join.MITER);
		touchPaint.setStrokeWidth(4f);		
	}
	
	@Override
	protected void onDraw(Canvas canvas) {
		canvas.drawPath(touchPath, touchPaint);
	}	
	
	@Override
	public boolean onTouchEvent(MotionEvent event) {

		float pointX = event.getX();
		float pointY = event.getY();

		switch (event.getAction()) {
		case MotionEvent.ACTION_DOWN:
			return true;
		case MotionEvent.ACTION_MOVE:
			touchPath.reset();
			touchPath.addCircle(pointX, pointY, 20, Path.Direction.CW);
			break;
		case MotionEvent.ACTION_UP:
			touchPath.reset();
			break;
		default:
			return false;
		}
		// Force view redraw.
		postInvalidate();
		return true;
	}	
}
