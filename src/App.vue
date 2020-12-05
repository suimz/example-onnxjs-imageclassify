<template>
  <div id="app">
    <div>
      <img id="img" src="/cat.jpg" :width="imgSize" :height="imgSize" />
    </div>
    <Button @click="onPredict" :disabled="isRunning || !isLoadModel">预测</Button>
    <div v-html="text"></div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator'
import { Tensor, InferenceSession } from 'onnxjs'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import { IMAGENET_CLASSES } from './imagenet'

@Component
export default class App extends Vue {
  // 图片尺寸,reset50模型限制
  imgSize = 224
  // 预测会话
  session = new InferenceSession()
  // 模型是否已加载
  isLoadModel = false
  // 正在预测
  isRunning = false
  // 结果
  text = ''

  mounted () {
    this.text = '正在加载模型文件...'
    const modelUrl = '/resnet50_8.onnx'
    // 加载模型到session中
    this.session.loadModel(modelUrl)
        .then(() => {
          this.isLoadModel = true
          this.text = '模型加载成功'
        })
        .catch(e => {
          console.error(e)
          this.text = '模型加载失败：' + e.message
        })
  }

  onPredict () {
    if (!this.isLoadModel) {
      alert('模型加载失败，无法预测')
      return
    } else if (this.isRunning) {
      alert('当前正在预测中')
      return
    }
    this.predict()
  }

  /**
   * 预测
   */
  predict () {
    this.isRunning = true
    this.text = '正在预测...'
    // 预处理数据
    const preprocessedData = this.preprocess()
    // 预测
    this.session.run([preprocessedData])
        .then((outputMap: ReadonlyMap<string, Tensor>) => this.printResult(outputMap))
        .catch(e => this.text = '预测失败：' + e.message)
        .finally(() => this.isRunning = false)
  }

  /**
   * 数据预处理
   */
  preprocess (): Tensor {
    // 获取出图片的数据
    const canvas = document.createElement('canvas')
    const context: any = canvas.getContext('2d')
    const img: any = document.getElementById('img')
    canvas.width = img.width;
    canvas.height = img.height;
    context.drawImage(img, 0, 0 );
    const imageData = context.getImageData(0, 0, img.width, img.height)
    const { data, width, height } = imageData
    // 数据及参数配置
    const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, height, width]);
    // 归一化 Normalize 0-255 to (-1)-1
    ops.divseq(dataFromImage, 128.0);
    ops.subseq(dataFromImage, 1.0);
    // 补位 Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
    ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2))
    ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1))
    ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0))
    // 创建张量
    const tensor: Tensor = new Tensor(new Float32Array(3 * width * height), 'float32', [1, 3, width, height]);
    (tensor.data as Float32Array).set(dataProcessed.data)
    return tensor
  }

  /**
   * 输出结果
   */
  printResult (outputMap: InferenceSession.OutputType) {
    // 结果集
    const data: ArrayLike<number> = outputMap.values().next().value.data
    if (!data || data.length === 0) {
      this.text = '没有预测出结果'
      return
    }
    // 转换成二维：[几率, 分类索引]
    const probsIndices = Array.from(data).map((prob, index) => { return [prob, index] })
    // 按几率对结果进行排序
    const sorted = probsIndices.sort(
        (a: Array<number>, b: Array<number>) => {
          if (a[0] < b[0]) {
            return -1
          }
          if (a[0] > b[0]) {
            return 1
          }
          return 0
        }
    ).reverse()
    // 取出前5个结果
    const results: Array<string> = []
    sorted.slice(0, 5).forEach((probIndex: any) => {
      // 取出索引
      const index = parseInt(probIndex[1], 10)
      // 根据索引取出对应的分类标签
      const iClass = (IMAGENET_CLASSES as any)[index]
      const name = iClass[1].replace(/_/g, ' ')
      const probability = probIndex[0]
      results.push(`${name}: ${Math.round(100 * probability)}%`)
    })
    this.text = results.join('<br/>')
  }

}
</script>
