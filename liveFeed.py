'''
*
* AUTHOR:   Andres Machado
            Mohammad Odeh (IST)
* ----------------------------------------------------------
* ----------------------------------------------------------
*
* RIGHT CLICK: Shutdown Program.
* LEFT CLICK: Toggle view.
'''

ver = "Pupil  Detection"

# Import necessary modules
import  numpy, cv2, argparse                                # Various Stuff
from    imutils.video.pivideostream import  PiVideoStream   # Import threaded PiCam module
from    imutils.video               import  FPS             # Benchmark FPS
from    time                        import  sleep           # Sleep for stability
from    threading                   import  Thread          # Used to thread processes
from    Queue                       import  Queue           # Used to queue input/output


# ************************************************************************
# =====================> DEFINE NECESSARY FUNCTIONS <=====================
# ************************************************************************

# *************************************
# Define right/left mouse click events
# *************************************
def control( event, x, y, flags, param ):
    global normalDisplay
    
    # Right button shuts down program
    if event == cv2.EVENT_RBUTTONDOWN:
        fps.stop()
        print( " [INFO] Elapsed time: {:.2f}".format(fps.elapsed()) )
        print( " [INFO] Approx. FPS : {:.2f}".format(fps.fps()) )

        # Do some shutdown clean up
        try:
            if ( t_scan4circles.isAlive() ):
                t_scan4circles.join(5.0)    # Terminate circle scanning thread
                print( " scan4circles: Terminated" )
                
            if ( t_procFrame.isAlive() ):
                t_procFrame.join(5.0)       # Terminate image processing thread
                print( " procFrame: Terminated" )

        except Exception as e:
            print( "Caught Error: %s" %str( type(e) ) )

        finally:
            stream.stop()                   # Stop capturing frames from stream
            cv2.destroyAllWindows()         # Close any open windows
            quit()                          # Shutdown python interpreter
        
    # Left button toggles display
    elif event == cv2.EVENT_LBUTTONDOWN:
        normalDisplay=not( normalDisplay )


# ****************************************************
# Define a placeholder function for trackbar. This is
# needed for the trackbars to function properly.
# ****************************************************
def placeholder( x ):
    pass


# ****************************************************
# Define function to apply required filters to image
# ****************************************************
def procFrame(bgr2gray, Q_procFrame):

    # Get trackbar position and reflect it threshold type and values
    threshType = cv2.getTrackbarPos( "Type:\n0.Binary\n1.BinaryInv\n2.Trunc\n3.2_0\n4.2_0Inv",
                                     "AI_View")
    thresholdVal    = cv2.getTrackbarPos( "thresholdVal", "AI_View")
    maxValue        = cv2.getTrackbarPos( "maxValue"    , "AI_View")

    # Dissolve noise while maintaining edge sharpness 
    bgr2gray = cv2.bilateralFilter( bgr2gray, 5, 17, 17 ) #( bgr2gray, 11, 17, 17 )
    bgr2gray = cv2.GaussianBlur( bgr2gray, (5, 5), 1 )

    # Threshold any color that is not black to white
    if threshType == 0:
        retval, thresholded = cv2.threshold( bgr2gray, thresholdVal, maxValue, cv2.THRESH_BINARY )
    elif threshType == 1:
        retval, thresholded = cv2.threshold( bgr2gray, thresholdVal, maxValue, cv2.THRESH_BINARY_INV )
    elif threshType == 2:
        retval, thresholded = cv2.threshold( bgr2gray, thresholdVal, maxValue, cv2.THRESH_TRUNC )
    elif threshType == 3:
        retval, thresholded = cv2.threshold( bgr2gray, thresholdVal, maxValue, cv2.THRESH_TOZERO )
    elif threshType == 4:
        retval, thresholded = cv2.threshold( bgr2gray, thresholdVal, maxValue, cv2.THRESH_TOZERO_INV )

    kernel = cv2.getStructuringElement( cv2.MORPH_RECT, ( 10, 10 ) )
    bgr2gray = cv2.erode( cv2.dilate( thresholded, kernel, iterations=1 ), kernel, iterations=1 )

    # Place processed image in queue for retrieval
    Q_procFrame.put(bgr2gray)


# ******************************************************
# Define a function to scan for circles from camera feed
# ******************************************************
def scan4circles( bgr2gray, frame, Q_scan4circles ):

    # Error handling in case a non-allowable integer is chosen (1)
    try:
        # Scan for circles
        circles = cv2.HoughCircles( bgr2gray, cv2.HOUGH_GRADIENT, dp, minDist,
                                    param1, param2, minRadius, maxRadius )

        # If circles are found draw them
        if circles is not None:
            circles = numpy.round( circles[0,:] ).astype( "int" )
            for ( x, y, r ) in circles:

                # Calculate radius in mm
                r_mm = r/pixelPerMetric
                
                # Draw circle
                rString = "Radius is %.3f" %r_mm 
                output = cv2.circle( frame, ( x, y ), r, ( 0, 255, 0 ), 4 )
                cv2.putText( output, rString, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2 )

    
        # If not within window resolution keep looking
        else:
            output = frame

        # Place output in queue for retrieval by main thread
        if Q_scan4circles.full() is False:
            Q_scan4circles.put( output )

    # Error handling in case a non-allowable integer is chosen (2)
    except Exception as instance:
        print( " Exception or Error Caught" )
        print( " Error Type %s" %str(type(instance)) )
        print( " Resetting Trackbars..." )

        # Reset trackbars
        cv2.createTrackbar( "dp"        , ver, 12   , 50 , placeholder ) #14
        cv2.createTrackbar( "minDist"   , ver, 570  , 750, placeholder )
        cv2.createTrackbar( "param1"    , ver, 124  , 750, placeholder ) #326
        cv2.createTrackbar( "param2"    , ver, 403  , 750, placeholder ) #231
        cv2.createTrackbar( "minRadius" , ver, 5    , 200, placeholder ) #1
        cv2.createTrackbar( "maxRadius" , ver, 20   , 250, placeholder )

        print( " Success" )

        # Exit function and re-loop
        return()


# ************************************************************************
# ===========================> SETUP PROGRAM <===========================
# ************************************************************************

# Setup camera
stream = PiVideoStream( resolution=(384, 288) ).start()
normalDisplay = True
sleep( 1.0 )

# Setup window and mouseCallback event
cv2.namedWindow( ver )
cv2.setMouseCallback( ver, control )

# Create a track bar for HoughCircles parameters
cv2.createTrackbar( "dp"        , ver, 12   , 50 , placeholder ) #14
cv2.createTrackbar( "minDist"   , ver, 570  , 750, placeholder )
cv2.createTrackbar( "param1"    , ver, 124  , 750, placeholder ) #326
cv2.createTrackbar( "param2"    , ver, 403  , 750, placeholder ) #231
cv2.createTrackbar( "minRadius" , ver, 5    , 200, placeholder ) #1
cv2.createTrackbar( "maxRadius" , ver, 20   , 250, placeholder )

# Setup window and trackbars for AI view
cv2.namedWindow( "AI_View" )

cv2.createTrackbar( "Type:\n0.Binary\n1.BinaryInv\n2.Trunc\n3.2_0\n4.2_0Inv",
                    "AI_View", 1, 4, placeholder )
cv2.createTrackbar( "thresholdVal", "AI_View", 50, 254, placeholder ) #45
cv2.createTrackbar( "maxValue", "AI_View", 255, 255, placeholder )

# Create a queue for retrieving data from thread
Q_procFrame     = Queue( maxsize=0 )
Q_scan4circles  = Queue( maxsize=0 )

# Start benchmark
print( " [INFO] Debug Mode: ON" )
fps = FPS().start()

# Emperically calculated ratio
pixelPerMetric = 20.0

# ************************************************************************
# =========================> MAKE IT ALL HAPPEN <=========================
# ************************************************************************

# Infinite loop
while True:
    
    # Get image from stream
    frame = stream.read()[36:252, 48:336]
    output = frame

    # Convert into grayscale because HoughCircle only accepts grayscale images
    bgr2gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

    # Start thread to process image and apply required filters to detect circles
    t_procFrame = Thread( target=procFrame, args=( bgr2gray, Q_procFrame ) )
    t_procFrame.start()

    # Check if queue has something available for retrieval
    if Q_procFrame.qsize() > 0:
        bgr2gray = Q_procFrame.get()

    # Get trackbar position and reflect it in HoughCircles parameters input
    dp = cv2.getTrackbarPos( "dp", ver )
    minDist = cv2.getTrackbarPos( "minDist", ver )
    param1 = cv2.getTrackbarPos( "param1", ver )
    param2 = cv2.getTrackbarPos( "param2", ver )
    minRadius = cv2.getTrackbarPos( "minRadius", ver )
    maxRadius = cv2.getTrackbarPos( "maxRadius", ver )

    # Start thread to scan for circles
    t_scan4circles = Thread( target=scan4circles, args=( bgr2gray, frame, Q_scan4circles ) )
    t_scan4circles.start()

    # Check if queue has something available for retrieval
    if Q_scan4circles.qsize() > 0:
        output = Q_scan4circles.get()

    # If debug flag is invoked
    fps.update()

    # Live feed display toggle
    if normalDisplay:
        cv2.imshow(ver, output)
        cv2.imshow( "AI_View", bgr2gray )
        key = cv2.waitKey(1) & 0xFF
    elif not(normalDisplay):
        cv2.imshow(ver, bgr2gray)
        key = cv2.waitKey(1) & 0xFF
