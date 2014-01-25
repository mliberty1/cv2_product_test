package com.jetperch.touchlatency;

import android.os.Bundle;
import android.app.Activity;

public class TouchLatencyActivity extends Activity {

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		TouchLatencyView tv = new TouchLatencyView(this);
		setContentView(tv);
	}

}
