// frontend/src/App.js
import React, { useState, useEffect, useRef } from 'react';
import './App.css'; 

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null); 
  const [models, setModels] = useState([]); // Backend'den çekilecek standart modeller
  const [selectedModel, setSelectedModel] = useState('yolov8n'); 
  const [trackers, setTrackers] = useState([]); 
  const [selectedTracker, setSelectedTracker] = useState('bytetrack');
  const [lineCoordinates, setLineCoordinates] = useState([{x: 0, y: 0}, {x: 0, y: 0}]); 
  const [isDrawing, setIsDrawing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState('Hazır');
  const [totalCount, setTotalCount] = useState(0);
  const [websocket, setWebsocket] = useState(null);
  const videoRef = useRef(null); 
  const videoCanvasRef = useRef(null); 
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 }); 
  const [startPoint, setStartPoint] = useState(null); 

  const [modelSelectionType, setModelSelectionType] = useState('standard_yolo');
  const [customModelUploadFile, setCustomModelUploadFile] = useState(null);
  const [uploadedCustomModelName, setUploadedCustomModelName] = useState('');

  // Seçilen modelin sınıf listesi
  const [modelClasses, setModelClasses] = useState([]); 
  // Kullanıcının seçtiği sınıf ID'leri
  const [selectedClassIds, setSelectedClassIds] = useState([]);

  // Dinamik tracker parametreleri
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.7);
  const [trackBufferFrames, setTrackBufferFrames] = useState(30);
  const [detectionHistory, setDetectionHistory] = useState([]);
  const [lastCounts, setLastCounts] = useState([]);


  const fetchModelsAndTrackers = () => {
    fetch('http://127.0.0.1:8000/models')
      .then(res => res.json())
      .then(data => {
        setModels(data.supported_models);
        console.log("[Frontend Debug] Standart modeller yüklendi:", data.supported_models);
      })
      .catch(err => console.error("[Frontend Hata] Standart modeller çekilemedi:", err));

    fetch('http://127.0.0.1:8000/trackers')
      .then(res => res.json())
      .then(data => {
        setTrackers(data.supported_trackers);
        console.log("[Frontend Debug] Takipçiler yüklendi:", data.supported_trackers);
      })
      .catch(err => console.error("[Frontend Hata] Takipçiler çekilemedi:", err));
  };

  const fetchLastCounts = () => {
    fetch('http://127.0.0.1:8000/last-10-counts')
      .then(res => res.json())
      .then(data => {
        setLastCounts(data);
        console.log("[Frontend Debug] Son 10 sayım verisi yüklendi:", data);
      })
      .catch(err => {
        console.error("[Frontend Hata] Son 10 sayım verisi alınamadı:", err);
      });
  };

  useEffect(() => {
    fetchModelsAndTrackers();
    fetchLastCounts();
    // diğer WebSocket ayarları...
  }, []);
  

  // Model değiştiğinde sınıf listesini çek
  useEffect(() => {
    if (selectedModel) {
      fetch(`http://127.0.0.1:8000/model-classes/${selectedModel}`)
        .then(res => res.json())
        .then(data => {
          setModelClasses(data.classes);
          // Yeni davranış: Varsayılan olarak hiçbir sınıf seçili gelmeyecek
          setSelectedClassIds([]); 
          console.log(`[Frontend Debug] Model '${selectedModel}' için sınıflar yüklendi:`, data.classes);
        })
        .catch(err => {
          console.error(`[Frontend Hata] Model '${selectedModel}' sınıfları çekilemedi:`, err);
          setModelClasses([]); 
          setSelectedClassIds([]);
        });
    } else {
      setModelClasses([]);
      setSelectedClassIds([]);
    }
  }, [selectedModel]); 


  useEffect(() => {
    fetchModelsAndTrackers(); 

    const ws = new WebSocket('ws://127.0.0.1:8000/ws/video-count');
    ws.onopen = () => {
      console.log('WebSocket Bağlantısı Kuruldu.');
      setWebsocket(ws);
    };
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('WebSocket Mesajı:', data);
      if (data.event === 'object_counted' || data.event === 'general_update' || data.event === 'video_ended') {
        setTotalCount(data.total_count);
        setProcessingStatus(`Toplam Sayım: ${data.total_count} (Son Olay: ${data.event})`);
        if (data.event === 'object_counted' && data.detail) {
          setDetectionHistory(prev => {
            const newHistory = [...prev, data.detail];
            return newHistory.slice(-10); // Son 10 taneyi tut
          });
        }
        if (data.event === 'video_ended' && data.path) {
          alert(`Video işleme tamamlandı ve kaydedildi: ${data.path}. Toplam Sayım: ${data.total_count}`);
          
          // İşlem bittikten sonra işlenmiş videoyu göstermek için URL'yi güncelle
          const processedVideoFilename = data.path.split('/').pop(); 
          const processedVideoRemoteUrl = `http://127.0.0.1:8000/processed-videos/${processedVideoFilename}`;
          setVideoUrl(processedVideoRemoteUrl); 
          setProcessingStatus(`Video işleme tamamlandı! Son Sayım: ${data.total_count}`); 
        }
      }
    };
    ws.onclose = () => {
      console.log('WebSocket Bağlantısı Kesildi.');
      setWebsocket(null);
      setProcessingStatus('WebSocket bağlantısı kesildi. Yeniden bağlanılıyor...');
      setTimeout(() => {
        const newWs = new WebSocket('ws://127.0.0.1:8000/ws/video-count');
        setWebsocket(newWs); 
      }, 8000);
    };
    ws.onerror = (error) => {
      console.error('WebSocket Hatası:', error);
      setProcessingStatus('WebSocket hatası oluştu.');
    };

    return () => {
      if (ws) ws.close();
    };
  }, []); 

  const handleVideoLoadedMetadata = () => {
    const video = videoRef.current;
    const canvas = videoCanvasRef.current; 
    
    if (!video || !canvas) {
        console.warn("[LoadedMetadata] Video veya Canvas referansı henüz mevcut değil, tekrar deneniyor...");
        setTimeout(() => handleVideoLoadedMetadata(), 50); 
        return; 
    }

    let actualVideoWidth = video.videoWidth;
    let actualVideoHeight = video.videoHeight;

    if (actualVideoWidth === 0 || actualVideoHeight === 0) {
      console.warn("[LoadedMetadata] Video boyutları henüz 0. Tekrar deneniyor...");
      setTimeout(() => {
        handleVideoLoadedMetadata(); 
      }, 100); 
      return; 
    }

    canvas.width = actualVideoWidth;
    canvas.height = actualVideoHeight;

    setVideoDimensions({ width: actualVideoWidth, height: actualVideoHeight });
    
    console.log(`[LoadedMetadata] Video Gerçek Boyutları: ${actualVideoWidth}x${actualVideoHeight}`);
    console.log(`[LoadedMetadata] Canvas Dahili Çözünürlük Ayarlandı: ${canvas.width}x${canvas.height}`);

    canvas.style.width = `${video.offsetWidth}px`;
    canvas.style.height = `${video.offsetHeight}px`;
  };

  const getMousePos = (canvas, evt) => {
    if (!canvas) {
      console.error("[getMousePos] Canvas referansı yok!");
      return { x: 0, y: 0 }; 
    }

    const rect = canvas.getBoundingClientRect(); 
    
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;  

    const mouseX = (evt.clientX - rect.left) * scaleX;
    const mouseY = (evt.clientY - rect.top) * scaleY;

    console.log(`[getMousePos] Mouse X,Y (Viewport): ${evt.clientX}, ${evt.clientY}`);
    console.log(`[getMousePos] Canvas Rect (Görüntülenen): ${JSON.stringify(rect)}`);
    console.log(`[getMousePos] Canvas Internal (Dahili Piksel): ${canvas.width}x${canvas.height}`);
    console.log(`[getMousePos] Scale X,Y: ${scaleX}, ${scaleY}`);
    console.log(`[getMousePos] Calculated Canvas X,Y (Hedef Piksel): ${mouseX}, ${mouseY}`);

    return {
      x: mouseX,
      y: mouseY
    };
  };

  const handleMouseDown = (e) => {
    if (!videoCanvasRef.current || videoDimensions.width === 0) {
      console.warn("[drawLines] Video ve canvas boyutları henüz yüklenmedi.");
      return; 
    }
    setIsDrawing(true);
    const pos = getMousePos(videoCanvasRef.current, e);
    setStartPoint(pos);
    setLineCoordinates([pos, pos]); 
  };

  const handleMouseMove = (e) => {
    if (!isDrawing || !videoCanvasRef.current || videoDimensions.width === 0) return; 
    const pos = getMousePos(videoCanvasRef.current, e);
    setLineCoordinates([startPoint, pos]); 
    drawLines(); 
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
    setStartPoint(null); 
    drawLines(); 
  };

  const drawLines = () => {
    const canvas = videoCanvasRef.current;
    if (!canvas || videoDimensions.width === 0) {
      console.warn("[drawLines] Canvas veya video boyutları hazır değil, çizilemiyor.");
      return;
    }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height); 

    if (lineCoordinates && lineCoordinates[0] && lineCoordinates[1]) {
      const p1 = lineCoordinates[0];
      const p2 = lineCoordinates[1];
      if (typeof p1.x === 'number' && typeof p1.y === 'number' &&
          typeof p2.x === 'number' && typeof p2.y === 'number') {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = 'yellow'; 
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = 'red';
        ctx.beginPath();
        ctx.arc(p1.x, p1.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(p2.x, p2.y, 5, 0, 2 * Math.PI);
        ctx.fill();
      } else {
        console.warn("[drawLines] Geçersiz çizgi koordinatları:", lineCoordinates);
      }
    }
  };

  useEffect(() => {
    drawLines();
  }, [lineCoordinates, videoDimensions]); 

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setVideoFile(file);
      const localVideoBlobUrl = URL.createObjectURL(file);
      setVideoUrl(localVideoBlobUrl);
      setProcessingStatus('Video yüklendi, işleme hazır.');
      setTotalCount(0); 
      setLineCoordinates([{x:0, y:0},{x:0, y:0}]); 
      setVideoDimensions({ width: 0, height: 0 }); 
    }
  };

  const handleCustomModelFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.name.endsWith(".pt")) {
      setCustomModelUploadFile(file);
      setUploadedCustomModelName(file.name); 
      console.log(`[Frontend] Yüklenecek özel model dosyası seçildi: ${file.name}`);
    } else {
      setCustomModelUploadFile(null);
      setUploadedCustomModelName('');
      alert("Lütfen sadece .pt uzantılı bir model dosyası seçin.");
    }
  };

  const uploadCustomModel = async () => {
    if (!customModelUploadFile) {
      alert("Lütfen önce yüklenecek bir özel model dosyası seçin.");
      return;
    }

    setProcessingStatus(`Model "${customModelUploadFile.name}" yükleniyor...`);
    const formData = new FormData();
    formData.append('model_file', customModelUploadFile);

    try {
      const response = await fetch('http://127.0.0.1:8000/upload-model/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Model yükleme hatası! Durum: ${response.status}, Mesaj: ${errorText}`);
      }

      const result = await response.json();
      alert(result.message);
      setProcessingStatus(`Model "${customModelUploadFile.name}" başarıyla yüklendi.`);
      setCustomModelUploadFile(null); 
      setSelectedModel(uploadedCustomModelName); 
      setModelSelectionType('custom_uploaded'); 

    } catch (error) {
      console.error('Model yükleme hatası:', error);
      setProcessingStatus(`Model yükleme hatası: ${error.message}`);
      alert(`Model yükleme sırasında bir hata oluştu: ${error.message}`);
    }
  };

  // Sınıf seçimi checkbox'larını yönetir
  const handleClassSelectionChange = (classId, isChecked) => {
    setSelectedClassIds(prevSelected => {
      if (isChecked) {
        return [...prevSelected, classId];
      } else {
        return prevSelected.filter(id => id !== classId);
      }
    });
  };

  // Tüm sınıfları seç
  const handleSelectAllClasses = () => {
    setSelectedClassIds(modelClasses.map(cls => cls.id));
  };

  // Tüm sınıfların seçimini kaldır
  const handleDeselectAllClasses = () => {
    setSelectedClassIds([]);
  };


  const startProcessing = async () => {
    if (!videoFile) {
      alert('Lütfen önce bir video dosyası seçin.');
      return;
    }

    if (modelSelectionType === 'custom_uploaded' && (!selectedModel || !selectedModel.includes(".pt"))) {
        alert("Lütfen özel bir model seçin veya yükleyin.");
        return;
    }

    setProcessingStatus('Video işleniyor...');
    setTotalCount(0); 

    const formData = new FormData();
    formData.append('video_file', videoFile);
    formData.append('model_name', selectedModel); 
    formData.append('tracker_name', selectedTracker);
    formData.append('line_coordinates', JSON.stringify([
      [Math.round(lineCoordinates[0].x), Math.round(lineCoordinates[0].y)],
      [Math.round(lineCoordinates[1].x), Math.round(lineCoordinates[1].y)]
    ]));
    // Seçilen sınıf ID'lerini JSON string olarak gönder
    formData.append('selected_class_ids', JSON.stringify(selectedClassIds));

    // Tracker parametrelerini gönder
    formData.append('conf_threshold', confThreshold);
    formData.append('iou_threshold', iouThreshold);
    formData.append('track_buffer_frames', trackBufferFrames);


    try {
      const response = await fetch('http://127.0.0.1:8000/process-video/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP hatası! Durum: ${response.status}, Mesaj: ${errorText}`);
      }

      setVideoUrl(URL.createObjectURL(await response.blob()));
      setProcessingStatus('Video akışı alınıyor...');

    } catch (error) {
      console.error('Video işleme hatası:', error);
      setProcessingStatus(`Hata: ${error.message}`);
      alert(`Video işleme sırasında bir hata oluştu: ${error.message}`);
    }
  };

  return (
    <div className="App">
      <h1>Object-Count Sistemi</h1>
  
      <div className="controls">
        <input type="file" accept="video/*" onChange={handleFileChange} />
        
        {/* Model Seçim Tipi */}
        <div className="model-selection-type">
          <label>
            <input 
              type="radio" 
              value="standard_yolo" 
              checked={modelSelectionType === 'standard_yolo'} 
              onChange={() => {
                setModelSelectionType('standard_yolo');
                setSelectedModel('yolov8n'); 
              }} 
            />
            Standart YOLO Modelleri
          </label>
          <label>
            <input 
              type="radio" 
              value="custom_uploaded" 
              checked={modelSelectionType === 'custom_uploaded'} 
              onChange={() => {
                setModelSelectionType('custom_uploaded');
                setSelectedModel(''); 
                setUploadedCustomModelName(''); 
                setCustomModelUploadFile(null); 
              }} 
            />
            Kendi Özel Modelimi Yükle
          </label>
        </div>

        {/* Standart Modeller Dropdown'ı */}
        {modelSelectionType === 'standard_yolo' && (
          <div className="model-dropdown">
            <label>Model Seçimi:</label>
            <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
              {models.length === 0 ? (
                <option value="">Standart Modeller Yükleniyor...</option>
              ) : (
                models.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))
              )}
            </select>
          </div>
        )}

        {/* Özel Model Yükleme ve Seçim Arayüzü */}
        {modelSelectionType === 'custom_uploaded' && (
          <div className="custom-model-section">
            <div className="model-upload">
              <label>Model Dosyası Seç (Yolov8 +) (.pt):</label>
              <input type="file" accept=".pt" onChange={handleCustomModelFileChange} />
              <button onClick={uploadCustomModel} disabled={!customModelUploadFile}>
                Modeli Yükle ve Seç
              </button>
              {(uploadedCustomModelName || selectedModel) && (modelSelectionType === 'custom_uploaded') && (
                 <p>Seçilen/Yüklü Model: <strong>{uploadedCustomModelName || selectedModel}</strong></p>
              )}
            </div>
          </div>
        )}

        {/* Sınıf Seçimi */}
        {modelClasses.length > 0 && (
          <div className="class-selection">
            <label>Sayılacak Sınıflar:</label>
            <div className="class-selection-buttons">
              <button onClick={handleSelectAllClasses}>Tümünü Seç</button>
              <button onClick={handleDeselectAllClasses}>Tümünü Kaldır</button>
            </div>
            <div className="class-checkboxes">
              {modelClasses.map(cls => (
                <label key={cls.id}>
                  <input 
                    type="checkbox" 
                    value={cls.id} 
                    checked={selectedClassIds.includes(cls.id)} 
                    onChange={(e) => handleClassSelectionChange(cls.id, e.target.checked)} 
                  />
                  {cls.name} ({cls.id})
                </label>
              ))}
            </div>
          </div>
        )}

        {/* Takip Algoritması */}
        <label>Takip Algoritması:</label>
        <select value={selectedTracker} onChange={(e) => setSelectedTracker(e.target.value)}>
          {trackers.map(tracker => (
            <option key={tracker} value={tracker}>{tracker}</option>
          ))}
        </select>

        {/* Dinamik Tracker Parametreleri */}
        <div className="tracker-params">
          <h3>Takip Parametreleri</h3>
          <label>
            Güven Eşiği (Conf Threshold):
            <input 
              type="number" 
              step="0.01" 
              min="0" 
              max="1" 
              value={confThreshold} 
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))} 
            />
          </label>
          <label>
            IoU Eşiği (IoU Threshold):
            <input 
              type="number" 
              step="0.01" 
              min="0" 
              max="1" 
              value={iouThreshold} 
              onChange={(e) => setIouThreshold(parseFloat(e.target.value))} 
            />
          </label>
          <label>
            Takip Tamponu (Track Buffer Frames):
            <input 
              type="number" 
              step="1" 
              min="1" 
              value={trackBufferFrames} 
              onChange={(e) => setTrackBufferFrames(parseInt(e.target.value))} 
            />
          </label>
        </div>


        <button onClick={startProcessing} disabled={!videoFile || processingStatus.startsWith('Video işleniyor')}>
          Videoyu İşle ve Sayımı Başlat
        </button>
      </div>
  
      <div className="video-container">
        {videoUrl && (
          <video
            ref={videoRef}
            src={videoUrl}
            controls={false}
            autoPlay
            muted
            loop
            onLoadedMetadata={handleVideoLoadedMetadata}
            className="responsive-video"
            style={{
              pointerEvents: 'none'
            }}
          >
            Tarayıcınız video etiketini desteklemiyor.
          </video>
        )}
        <canvas
          ref={videoCanvasRef}
          className="drawing-canvas"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{
            display: videoDimensions.width > 0 ? 'block' : 'none',
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            border: '2px dashed blue',
            zIndex: 10,
            cursor: 'crosshair',
          }}
        ></canvas>
      </div>
  
      <div className="status-area">
        <p>İşlem Durumu: <strong>{processingStatus}</strong></p>
        <p>Canlı Sayım: <strong>{totalCount}</strong></p>
        <p>Çizgi Koordinatları: {JSON.stringify(lineCoordinates)}</p>
      </div>
  
      <div className="last-counts-section">
        <h2>Son 10 Sayım</h2>
        {Array.isArray(lastCounts) && lastCounts.length === 0 ? (
          <p>Veri bulunamadı</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Video Adı</th>
                <th>Model</th>
                <th>Tracker</th>
                <th>Sayım</th>
                <th>Başlangıç</th>
                <th>Bitiş</th>
                <th>Video</th>
              </tr>
            </thead>
            <tbody>
              {lastCounts.map(record => (
                <tr key={record.id}>
                  <td>{record.id}</td>
                  <td>{record.video_name}</td>
                  <td>{record.model_used}</td>
                  <td>{record.tracker_used}</td>
                  <td>{record.final_count}</td>
                  <td>{record.start_time}</td>
                  <td>{record.end_time}</td>
                  <td>
                    {record.processed_video_path ? (
                      <a
                        href={`http://127.0.0.1:8000/processed-videos/${record.processed_video_path.split('/').pop()}`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        İzle
                      </a>
                    ) : (
                      "Henüz mevcut değil"
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>  
  );
}

export default App;